"""
JAX translation of MLLeafBoundaryLayerMod Fortran module.

Leaf boundary layer conductance for heat, water vapour, and CO2.

Original Fortran module: MLLeafBoundaryLayerMod
Fortran lines 1-140
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from clm_src_main.abortutils import endrun                           # noqa: F401
from clm_src_main.clm_varcon import tfrz, grav                      # noqa: F401
from clm_src_main.PatchType import patch                            # noqa: F401
from clm_src_main.pftconMod import pftcon                           # noqa: F401
from multilayer_canopy.MLclm_varcon import visc0, dh0, dv0, dc0, gb_factor, gbh_min  # noqa: F401
from multilayer_canopy.MLclm_varctl import gb_type                       # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type           # noqa: F401


def LeafBoundaryLayer(
    num_filter: int,
    filter_patch: Sequence[int],
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf boundary layer conductance for heat, water vapour,
    and CO2.

    Mirrors Fortran subroutine ``LeafBoundaryLayer`` (lines 22-140).

    Reference: Bonan (2019) *Climate Change and Terrestrial Ecosystem
    Modeling*, Chapter 10.

    Molecular diffusivities are corrected for temperature and pressure
    relative to standard conditions (Fortran lines 72-76):

    .. code-block:: none

        fac  = 101325 / pref * (tref / tfrz)^1.81
        visc = visc0 * fac
        dh   = dh0 * fac
        dv   = dv0 * fac
        dc   = dc0 * fac

    For each leaf layer with ``dpai > 0``, dimensionless numbers are
    computed (Fortran lines 80-83):

    .. code-block:: none

        Re = wind * dleaf / visc
        Pr = visc / dh
        Gr = grav * dleaf^3 * max(tleaf - tair, 0) / (tair * visc^2)

    Three Nusselt-number regimes are computed (Fortran lines 85-101):

    - **Laminar forced** (Fortran lines 87-90):
      ``Nu = gb_factor * 0.66 * Pr^0.33 * Re^0.5``
    - **Turbulent forced** (Fortran lines 92-96):
      ``Nu = gb_factor * 0.036 * Pr^0.33 * Re^0.8``
    - **Free convection** (Fortran lines 98-101):
      ``Nu = 0.54 * Pr^0.25 * Gr^0.25``

    Conductances from each regime are:

    .. code-block:: none

        gbh = (dh * Nu / dleaf) * rhomol    [mol/m2/s]
        gbv = gbh * (dv/dh)^exponent
        gbc = gbh * (dc/dh)^exponent

    Exponent is 0.67 for forced convection and 0.75 for free convection.
    ``gbh_lam`` and ``gbh_turb`` are floored at ``gbh_min``.

    The ``gb_type`` switch selects which regimes are combined
    (Fortran lines 103-121):

    - **1** — laminar only.
    - **2** — max of laminar and turbulent.
    - **3** — max of laminar and turbulent, plus free convection.

    Layers with ``dpai == 0`` receive all conductances set to zero.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; ``gbh_leaf``, ``gbv_leaf``,
            and ``gbc_leaf`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dleaf = pftcon.dleaf

    gbh = mlcanopy_inst.gbh_leaf
    gbv = mlcanopy_inst.gbv_leaf
    gbc = mlcanopy_inst.gbc_leaf

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])

        # Diffusivity correction for temperature and pressure — Fortran lines 72-76
        pref_p  = float(mlcanopy_inst.pref_forcing[p])
        tref_p  = float(mlcanopy_inst.tref_forcing[p])
        fac     = 101325.0 / pref_p * (tref_p / tfrz) ** 1.81
        visc    = visc0 * fac
        dh      = dh0  * fac
        dv      = dv0  * fac
        dc      = dc0  * fac

        rhomol_p = float(mlcanopy_inst.rhomol_forcing[p])
        dl       = float(dleaf[pft])                   # leaf dimension (m)
        dv_dh    = dv / dh
        dc_dh    = dc / dh

        # Pre-extract per-layer inputs as numpy — one JAX sync each
        _ncan_p  = int(mlcanopy_inst.ncan_canopy[p])
        _dpai_p  = np.asarray(mlcanopy_inst.dpai_profile[p])
        _wind_p  = np.asarray(mlcanopy_inst.wind_profile[p])
        _tair_p  = np.asarray(mlcanopy_inst.tair_profile[p])
        _tleaf_p = np.asarray(mlcanopy_inst.tleaf_leaf[p, :, il])

        # Allocate output numpy arrays
        _sl = slice(1, _ncan_p + 1)
        _gbh_new = np.zeros(_ncan_p + 2)
        _gbv_new = np.zeros(_ncan_p + 2)
        _gbc_new = np.zeros(_ncan_p + 2)

        for ic in range(1, _ncan_p + 1):

            if _dpai_p[ic] > 0.0:

                wind_ic  = float(_wind_p[ic])
                tair_ic  = float(_tair_p[ic])
                tleaf_ic = float(_tleaf_p[ic])

                # Dimensionless numbers — Fortran lines 80-83
                re = wind_ic * dl / visc
                pr = visc / dh
                gr = (grav * dl ** 3
                      * max(tleaf_ic - tair_ic, 0.0)
                      / (tair_ic * visc * visc))

                # (i) Laminar forced convection — Fortran lines 87-90
                nu        = gb_factor * 0.66 * pr ** 0.33 * re ** 0.5
                gbh_lam   = max((dh * nu / dl) * rhomol_p, gbh_min)
                gbv_lam   = gbh_lam * dv_dh ** 0.67
                gbc_lam   = gbh_lam * dc_dh ** 0.67

                # (ii) Turbulent forced convection — Fortran lines 92-96
                nu        = gb_factor * 0.036 * pr ** 0.33 * re ** 0.8
                gbh_turb  = max((dh * nu / dl) * rhomol_p, gbh_min)
                gbv_turb  = gbh_turb * dv_dh ** 0.67
                gbc_turb  = gbh_turb * dc_dh ** 0.67

                # (iii) Free convection — Fortran lines 98-101
                nu        = 0.54 * pr ** 0.25 * gr ** 0.25
                gbh_free  = (dh * nu / dl) * rhomol_p
                gbv_free  = gbh_free * dv_dh ** 0.75
                gbc_free  = gbh_free * dc_dh ** 0.75

                # Select flow regime — Fortran lines 103-121
                if gb_type == 1:
                    gbh_val = gbh_lam
                    gbv_val = gbv_lam
                    gbc_val = gbc_lam

                elif gb_type == 2:
                    gbh_val = max(gbh_lam, gbh_turb)
                    gbv_val = max(gbv_lam, gbv_turb)
                    gbc_val = max(gbc_lam, gbc_turb)

                elif gb_type == 3:
                    gbh_val = max(gbh_lam, gbh_turb) + gbh_free
                    gbv_val = max(gbv_lam, gbv_turb) + gbv_free
                    gbc_val = max(gbc_lam, gbc_turb) + gbc_free

                else:
                    endrun(msg=' ERROR: LeafBoundaryLayer: gb_type not valid')
                    gbh_val = gbv_val = gbc_val = 0.0    # Unreachable

            else:                                        # Fortran lines 123-125
                gbh_val = 0.0
                gbv_val = 0.0
                gbc_val = 0.0

            _gbh_new[ic] = gbh_val
            _gbv_new[ic] = gbv_val
            _gbc_new[ic] = gbc_val

        # Batch write-back — one JAX operation per field per patch
        gbh = gbh.at[p, _sl, il].set(jnp.array(_gbh_new[_sl]))
        gbv = gbv.at[p, _sl, il].set(jnp.array(_gbv_new[_sl]))
        gbc = gbc.at[p, _sl, il].set(jnp.array(_gbc_new[_sl]))

    return mlcanopy_inst._replace(
        gbh_leaf = gbh,
        gbv_leaf = gbv,
        gbc_leaf = gbc,
    )