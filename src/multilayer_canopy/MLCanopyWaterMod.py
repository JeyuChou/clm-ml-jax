"""
JAX translation of MLCanopyWaterMod Fortran module.

Canopy water interception, evaporation, and wetted fraction.
Provides three public routines:

- :func:`CanopyWettedFraction`: wetted and dry fractions of each layer.
- :func:`CanopyInterception`: interception and throughfall.
- :func:`CanopyEvaporation`: update intercepted water for evaporation/dew.

Original Fortran module: MLCanopyWaterMod
Fortran lines 1-215
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from multilayer_canopy.MLclm_varcon import (                              # noqa: F401
    dewmx,
    maximum_leaf_wetted_fraction,
    fwet_exponent,
    interception_fraction,
    mmh2o,
)
from multilayer_canopy.MLclm_varctl import dtime_ml                      # noqa: F401
from multilayer_canopy.MLclm_varpar import isun, isha                    # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type           # noqa: F401


# ---------------------------------------------------------------------------
# Public: wetted fraction of canopy
# ---------------------------------------------------------------------------

def CanopyWettedFraction(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate the wetted fraction and the green-dry fraction of each
    canopy layer.

    Mirrors Fortran subroutine ``CanopyWettedFraction`` (lines 31-80).

    For layers with ``dpai > 0`` (Fortran lines 61-73):

    .. code-block:: none

        h2ocanmx = dewmx * dpai(ic)
        fwet     = min(max(h2ocan/h2ocanmx, 0)^fwet_exponent,
                       maximum_leaf_wetted_fraction)
        fdry     = (1 - fwet) * dlai / dpai

    Layers with ``dpai == 0`` receive ``fwet = fdry = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``fwet_profile`` and
            ``fdry_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    fwet = mlcanopy_inst.fwet_profile
    fdry = mlcanopy_inst.fdry_profile

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])
        _ncan = int(mlcanopy_inst.ncan_canopy[p])

        # Pre-extract as numpy — one JAX sync per array
        _dpai   = np.asarray(mlcanopy_inst.dpai_profile[p])
        _h2ocan = np.asarray(mlcanopy_inst.h2ocan_profile[p])
        _dlai   = np.asarray(mlcanopy_inst.dlai_profile[p])

        ics      = np.arange(1, _ncan + 1)
        dpai_v   = _dpai[ics]
        h2ocan_v = _h2ocan[ics]
        dlai_v   = _dlai[ics]

        has_pai  = dpai_v > 0.0
        dpai_safe  = np.where(has_pai, dpai_v, 1.0)   # avoid div-by-zero
        h2ocanmx   = dewmx * dpai_safe
        fwet_v = np.where(
            has_pai,
            np.minimum(
                np.maximum(h2ocan_v / h2ocanmx, 0.0) ** fwet_exponent,
                maximum_leaf_wetted_fraction,
            ),
            0.0,
        )
        fdry_v = np.where(
            has_pai,
            (1.0 - fwet_v) * dlai_v / dpai_safe,
            0.0,
        )

        _fwet_new        = np.zeros(_ncan + 2)
        _fdry_new        = np.zeros(_ncan + 2)
        _fwet_new[ics]   = fwet_v
        _fdry_new[ics]   = fdry_v

        _sl = slice(1, _ncan + 1)
        fwet = fwet.at[p, _sl].set(jnp.array(_fwet_new[_sl]))
        fdry = fdry.at[p, _sl].set(jnp.array(_fdry_new[_sl]))

    return mlcanopy_inst._replace(
        fwet_profile = fwet,
        fdry_profile = fdry,
    )


# ---------------------------------------------------------------------------
# Public: interception and throughfall
# ---------------------------------------------------------------------------

def CanopyInterception(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate canopy interception and throughfall.

    Mirrors Fortran subroutine ``CanopyInterception`` (lines 82-165).

    **Patch-level calculations** (Fortran lines 120-133):

    .. code-block:: none

        fracrain = qflx_rain / (qflx_snow + qflx_rain)  [or 0 if no precip]
        fracsnow = qflx_snow / (qflx_snow + qflx_rain)
        fpi      = interception_fraction * tanh(lai + sai)
        qflx_through_rain = qflx_rain * (1 - fpi)
        qflx_through_snow = qflx_snow * (1 - fpi)
        qflx_intr         = (qflx_snow + qflx_rain) * fpi

    Intercepted precipitation is distributed equally across all layers
    with ``dpai > 0`` (count = ``n``).  Per-layer water balance
    (Fortran lines 141-153):

    .. code-block:: none

        h2ocan(ic) = h2ocan_bef(ic) + qflx_intr * dtime / n
        xrun       = (h2ocan(ic) - h2ocanmx) / dtime
        if xrun > 0:
            qflx_candrip += xrun
            h2ocan(ic)    = h2ocanmx

    Total throughfall onto ground (Fortran lines 157-158):

    .. code-block:: none

        qflx_tflrain = qflx_through_rain + qflx_candrip * fracrain
        qflx_tflsnow = qflx_through_snow + qflx_candrip * fracsnow

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``h2ocan_profile``,
            ``qflx_intr_canopy``, ``qflx_tflrain_canopy``, and
            ``qflx_tflsnow_canopy`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dtime: float = float(dtime_ml)

    h2ocan      = mlcanopy_inst.h2ocan_profile
    qflx_intr   = mlcanopy_inst.qflx_intr_canopy
    qflx_tflrain = mlcanopy_inst.qflx_tflrain_canopy
    qflx_tflsnow = mlcanopy_inst.qflx_tflsnow_canopy

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])

        rain_p = float(mlcanopy_inst.qflx_rain_forcing[p])
        snow_p = float(mlcanopy_inst.qflx_snow_forcing[p])

        # Rain/snow fractions — Fortran lines 120-125
        total_precip = snow_p + rain_p
        if total_precip > 0.0:
            fracrain = rain_p / total_precip
            fracsnow = snow_p / total_precip
        else:
            fracrain = 0.0
            fracsnow = 0.0

        # Intercepted fraction (CLM5 form) — Fortran line 127
        lai_p = float(mlcanopy_inst.lai_canopy[p])
        sai_p = float(mlcanopy_inst.sai_canopy[p])
        fpi = interception_fraction * math.tanh(lai_p + sai_p)

        # Direct throughfall — Fortran lines 129-130
        qflx_through_rain = rain_p * (1.0 - fpi)
        qflx_through_snow = snow_p * (1.0 - fpi)

        # Intercepted precipitation — Fortran line 132
        qflx_intr_p = total_precip * fpi
        qflx_intr = qflx_intr.at[p].set(qflx_intr_p)

        # Count layers with dpai > 0 and pre-extract arrays — Fortran lines 134-136
        ncan_p  = int(mlcanopy_inst.ncan_canopy[p])
        _dpai_p = np.asarray(mlcanopy_inst.dpai_profile[p])
        _h2ocan_bef_p = np.asarray(mlcanopy_inst.h2ocan_bef_profile[p])
        ics     = np.arange(1, ncan_p + 1)
        dpai_v  = _dpai_p[ics]
        has_pai = dpai_v > 0.0
        n       = int(np.sum(has_pai))

        # Per-layer water balance — vectorised — Fortran lines 138-153
        h2ocan_bef_v = _h2ocan_bef_p[ics]
        dpai_safe    = np.where(has_pai, dpai_v, 1.0)
        h2ocanmx_v   = dewmx * dpai_safe
        add_per_layer = (qflx_intr_p * dtime / float(n)) if n > 0 else 0.0
        h2ocan_v     = np.where(has_pai, h2ocan_bef_v + add_per_layer, 0.0)
        xrun_v       = np.where(has_pai, (h2ocan_v - h2ocanmx_v) / dtime, 0.0)
        drip_mask    = xrun_v > 0.0
        qflx_candrip = float(np.sum(np.where(drip_mask, xrun_v, 0.0)))
        h2ocan_v     = np.where(drip_mask, h2ocanmx_v, h2ocan_v)

        _h2ocan_new      = np.zeros(ncan_p + 2)
        _h2ocan_new[ics] = h2ocan_v
        _sl = slice(1, ncan_p + 1)
        h2ocan = h2ocan.at[p, _sl].set(jnp.array(_h2ocan_new[_sl]))

        # Total throughfall — Fortran lines 157-158
        qflx_tflrain = qflx_tflrain.at[p].set(
            qflx_through_rain + qflx_candrip * fracrain
        )
        qflx_tflsnow = qflx_tflsnow.at[p].set(
            qflx_through_snow + qflx_candrip * fracsnow
        )

    return mlcanopy_inst._replace(
        h2ocan_profile       = h2ocan,
        qflx_intr_canopy     = qflx_intr,
        qflx_tflrain_canopy  = qflx_tflrain,
        qflx_tflsnow_canopy  = qflx_tflsnow,
    )


# ---------------------------------------------------------------------------
# Public: update intercepted water for evaporation and dew
# ---------------------------------------------------------------------------

def CanopyEvaporation(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Update canopy intercepted water for evaporation and dew deposition.

    Mirrors Fortran subroutine ``CanopyEvaporation`` (lines 167-215).

    For each layer with ``dpai > 0``, dew deposition is applied first
    (negative evaporation/transpiration flux = condensation), then
    positive evaporation is removed.  Sunlit and shaded leaf populations
    are treated separately and weighted by ``fracsun`` and
    ``(1 - fracsun)`` (Fortran lines 195-210):

    .. code-block:: none

        # --- Dew (add negative fluxes) ---
        dew_sun = (evleaf_sun + trleaf_sun) * fracsun * dpai * mmh2o * dtime
        if dew_sun < 0:  h2ocan -= dew_sun

        dew_sha = (evleaf_sha + trleaf_sha) * (1-fracsun) * dpai * mmh2o * dtime
        if dew_sha < 0:  h2ocan -= dew_sha

        # --- Evaporation (remove positive fluxes) ---
        if evleaf_sun > 0:  h2ocan -= evleaf_sun * fracsun * dpai * mmh2o * dtime
        if evleaf_sha > 0:  h2ocan -= evleaf_sha * (1-fracsun) * dpai * mmh2o * dtime

    The commented-out floor ``h2ocan = max(0, h2ocan)`` is preserved
    as a comment.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``h2ocan_profile`` is updated
            in-place (input/output field).

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dtime: float = float(dtime_ml)

    h2ocan = mlcanopy_inst.h2ocan_profile

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])
        _ncan = int(mlcanopy_inst.ncan_canopy[p])

        # Pre-extract as numpy — one sync per array
        _dpai    = np.asarray(mlcanopy_inst.dpai_profile[p])
        _fracsun = np.asarray(mlcanopy_inst.fracsun_profile[p])
        _h2ocan  = np.asarray(h2ocan[p])
        _evleaf  = np.asarray(mlcanopy_inst.evleaf_leaf[p])   # shape (nlevmlcan+1, nleaf+1)
        _trleaf  = np.asarray(mlcanopy_inst.trleaf_leaf[p])

        ics      = np.arange(1, _ncan + 1)
        dpai_v   = _dpai[ics]
        fs_v     = _fracsun[ics]
        h2o_v    = _h2ocan[ics]
        has_pai  = dpai_v > 0.0
        dpai_safe = np.where(has_pai, dpai_v, 0.0)
        factor_v = dpai_safe * mmh2o * dtime

        evleaf_sun_v = _evleaf[ics, isun]
        trleaf_sun_v = _trleaf[ics, isun]
        evleaf_sha_v = _evleaf[ics, isha]
        trleaf_sha_v = _trleaf[ics, isha]

        # Dew (negative flux adds to h2ocan) — Fortran lines 195-203
        dew_sun_v = (evleaf_sun_v + trleaf_sun_v) * fs_v * factor_v
        dew_sha_v = (evleaf_sha_v + trleaf_sha_v) * (1.0 - fs_v) * factor_v
        h2o_v = h2o_v - np.where(dew_sun_v < 0.0, dew_sun_v, 0.0)
        h2o_v = h2o_v - np.where(dew_sha_v < 0.0, dew_sha_v, 0.0)

        # Evaporation (positive flux removes from h2ocan) — Fortran lines 205-211
        h2o_v = h2o_v - np.where(evleaf_sun_v > 0.0, evleaf_sun_v * fs_v * factor_v, 0.0)
        h2o_v = h2o_v - np.where(evleaf_sha_v > 0.0, evleaf_sha_v * (1.0 - fs_v) * factor_v, 0.0)

        _h2ocan_new      = np.zeros(_ncan + 2)
        _h2ocan_new[ics] = np.where(has_pai, h2o_v, _h2ocan[ics])
        _sl = slice(1, _ncan + 1)
        h2ocan = h2ocan.at[p, _sl].set(jnp.array(_h2ocan_new[_sl]))

    return mlcanopy_inst._replace(h2ocan_profile = h2ocan)