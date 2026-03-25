"""
JAX translation of MLCanopyNitrogenProfileMod Fortran module.

Canopy profile of nitrogen and photosynthetic capacity.

Original Fortran module: MLCanopyNitrogenProfileMod
Fortran lines 1-175
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from clm_src_main.abortutils import endrun                           # noqa: F401
from clm_src_main.clm_varcon import tfrz                            # noqa: F401
from clm_src_main.PatchType import patch                            # noqa: F401
from clm_src_main.pftconMod import pftcon                           # noqa: F401
from multilayer_canopy.MLclm_varcon import (                             # noqa: F401
    jmax25_to_vcmax25_noacclim, jmax25_to_vcmax25_acclim,
    rd25_to_vcmax25_c3, rd25_to_vcmax25_c4, kp25_to_vcmax25_c4,
)
from multilayer_canopy.MLclm_varctl import acclim_type, kn_val, leaf_optics_type  # noqa: F401
from multilayer_canopy.MLclm_varpar import isun, isha                    # noqa: F401
from multilayer_canopy.MLpftconMod import MLpftcon                       # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type           # noqa: F401


def CanopyNitrogenProfile(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate the canopy profile of nitrogen and photosynthetic capacity.

    Mirrors Fortran subroutine ``CanopyNitrogenProfile`` (lines 22-175).

    References:
    Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A1)-(A2).
    Bonan et al. (2021) *Agric. For. Met.*, 306, 108435,
    supplemental eqs. (6)-(12).

    **Canopy-top parameters** (Fortran lines 80-100):

    - ``vcmax25top`` from PFT lookup table.
    - ``jmax25_to_vcmax25`` selected by ``acclim_type``; for
      ``acclim_type == 1`` the ratio is temperature-dependent:
      ``2.59 - 0.035 * clamp(tacclim - tfrz, 11, 35)``.
    - C3: ``jmax25top = ratio * vcmax25top``,
      ``rd25top = rd25_to_vcmax25_c3 * vcmax25top``, ``kp25top = 0``.
    - C4: ``jmax25top = 0``,
      ``rd25top = rd25_to_vcmax25_c4 * vcmax25top``,
      ``kp25top = kp25_to_vcmax25_c4 * vcmax25top``.

    **Leaf nitrogen decay coefficient** (Fortran lines 102-110):

    .. code-block:: none

        kn_val < 0: kn = exp(0.00963 * vcmax25top - 2.43)
        kn_val > 0: kn = kn_val

    **Per-layer nitrogen scaling** (Fortran lines 113-148):
    Iterates from top to bottom (``ic = ncan, ..., 1``).

    .. code-block:: none

        fn     = exp(-kn*pai_above) * (1 - exp(-kn*dpai)) / kn

    For ``leaf_optics_type == 0`` (Bonan et al. 2021, eqs. 8-12):

    .. code-block:: none

        fn_sun = clump / (kn + kb*clump) * exp(-kn*pai_above)
                 * tbi * (1 - exp(-(kn + kb*clump)*dpai))
        fn_sha = fn - fn_sun
        nscale_sun = fn_sun / (fracsun * dpai)
        nscale_sha = fn_sha / ((1-fracsun) * dpai)

    For ``leaf_optics_type == 1``:

    .. code-block:: none

        nscale_sun = nscale_sha = fn / dpai

    All four leaf parameters (``vcmax25``, ``jmax25``, ``rd25``,
    ``kp25``) are scaled proportionally to ``nscale``.

    **Layer weighted mean** (Fortran lines 150-154):

    .. code-block:: none

        vcmax25_profile(ic) = vcmax25_leaf(ic,isun)*fracsun
                            + vcmax25_leaf(ic,isha)*(1-fracsun)

    **Conservation check** (Fortran lines 156-160):
    Numerical sum of ``vcmax25_profile * dpai`` over all layers must
    equal the analytical integral
    ``vcmax25top * (1 - exp(-kn*(lai+sai))) / kn``
    to within 1e-6 umol/m2/s.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``vcmax25_leaf``,
            ``jmax25_leaf``, ``rd25_leaf``, ``kp25_leaf``,
            ``vcmax25_profile``, ``jmax25_profile``, ``rd25_profile``,
            and ``kp25_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    c3psn = pftcon.c3psn

    vcmax25_leaf    = mlcanopy_inst.vcmax25_leaf
    jmax25_leaf     = mlcanopy_inst.jmax25_leaf
    rd25_leaf       = mlcanopy_inst.rd25_leaf
    kp25_leaf       = mlcanopy_inst.kp25_leaf
    vcmax25_profile = mlcanopy_inst.vcmax25_profile
    jmax25_profile  = mlcanopy_inst.jmax25_profile
    rd25_profile    = mlcanopy_inst.rd25_profile
    kp25_profile    = mlcanopy_inst.kp25_profile

    for fp in range(1, num_filter + 1):            # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])
        is_c3 = round(float(c3psn[pft])) == 1

        # Canopy-top photosynthetic parameters — Fortran lines 80-96
        vcmax25top = float(MLpftcon.vcmaxpft[pft])

        if acclim_type == 0:
            j2v = float(jmax25_to_vcmax25_noacclim)
        elif acclim_type == 1:
            ta_c = min(max(float(mlcanopy_inst.tacclim_forcing[p]) - tfrz,
                           11.0), 35.0)
            j2v  = 2.59 - 0.035 * ta_c               # Fortran line 87
        else:
            endrun(msg=' ERROR: CanopyNitrogenProfile: acclim_type not valid')
            j2v = 0.0    # Unreachable

        if is_c3:                                  # Fortran lines 90-93
            jmax25top = j2v            * vcmax25top
            rd25top   = rd25_to_vcmax25_c3 * vcmax25top
            kp25top   = 0.0
        else:                                      # Fortran lines 94-97
            jmax25top = 0.0
            rd25top   = rd25_to_vcmax25_c4 * vcmax25top
            kp25top   = kp25_to_vcmax25_c4 * vcmax25top

        # Nitrogen decay coefficient — Fortran lines 102-108
        if kn_val < 0.0:
            kn = math.exp(0.00963 * vcmax25top - 2.43)
        elif kn_val > 0.0:
            kn = float(kn_val)
        else:
            endrun(msg='ERROR: CanopyNitrogenProfile: incorrect Kn')
            kn = 0.0    # Unreachable

        clump = float(MLpftcon.clump_fac[pft])
        _ncan = int(mlcanopy_inst.ncan_canopy[p])
        lai_p = float(mlcanopy_inst.lai_canopy[p])
        sai_p = float(mlcanopy_inst.sai_canopy[p])

        # Pre-extract per-layer arrays as numpy — one sync each
        _dpai    = np.asarray(mlcanopy_inst.dpai_profile[p])   # shape (nlevmlcan+1,)
        _kb      = np.asarray(mlcanopy_inst.kb_profile[p])
        _tbi     = np.asarray(mlcanopy_inst.tbi_profile[p])
        _fracsun = np.asarray(mlcanopy_inst.fracsun_profile[p])

        # Iteration order: top → bottom = ic = ncan, ncan-1, ..., 1
        # ics_top_to_bot[j] = ncan - j  (j=0 is top layer)
        ics = np.arange(_ncan, 0, -1)          # [ncan, ncan-1, ..., 1]
        dpai_v   = _dpai[ics]                  # dpai in top→bottom order
        kb_v     = _kb[ics]
        tbi_v    = _tbi[ics]
        fs_v     = _fracsun[ics]

        # pai_above[j] = cumulative dpai of layers above layer ics[j]
        # = sum(dpai_v[0:j])  →  shifted cumsum — Fortran line 148
        pai_above_v = np.concatenate([[0.0], np.cumsum(dpai_v[:-1])])

        has_pai = dpai_v > 0.0
        dpai_safe = np.where(has_pai, dpai_v, 1.0)       # avoid /0

        # Integrated nitrogen factor — Fortran line 130
        exp_kn_above = np.exp(-kn * pai_above_v)
        fn = exp_kn_above * (1.0 - np.exp(-kn * dpai_v)) / kn
        fn = np.where(has_pai, fn, 0.0)

        if leaf_optics_type == 0:              # Fortran lines 132-139
            denom = kn + kb_v * clump
            denom_safe = np.where(has_pai, denom, 1.0)
            fn_sun = (clump / denom_safe * exp_kn_above * tbi_v
                      * (1.0 - np.exp(-denom_safe * dpai_v)))
            fn_sun = np.where(has_pai, fn_sun, 0.0)
            fn_sha = fn - fn_sun
            fs_safe  = np.where(fs_v > 0.0, fs_v, 1.0)
            fsha_safe = np.where((1.0 - fs_v) > 0.0, (1.0 - fs_v), 1.0)
            nscale_sun = np.where(has_pai, fn_sun / (fs_safe  * dpai_safe), 0.0)
            nscale_sha = np.where(has_pai, fn_sha / (fsha_safe * dpai_safe), 0.0)

        elif leaf_optics_type == 1:            # Fortran lines 140-141
            nscale = np.where(has_pai, fn / dpai_safe, 0.0)
            nscale_sun = nscale
            nscale_sha = nscale

        else:
            nscale_sun = np.zeros(_ncan)
            nscale_sha = np.zeros(_ncan)

        # Scale leaf parameters — Fortran lines 143-146 (vectorized)
        vcmax25_sun_v = np.where(has_pai, vcmax25top * nscale_sun, 0.0)
        vcmax25_sha_v = np.where(has_pai, vcmax25top * nscale_sha, 0.0)
        jmax25_sun_v  = np.where(has_pai, jmax25top  * nscale_sun, 0.0)
        jmax25_sha_v  = np.where(has_pai, jmax25top  * nscale_sha, 0.0)
        rd25_sun_v    = np.where(has_pai, rd25top    * nscale_sun, 0.0)
        rd25_sha_v    = np.where(has_pai, rd25top    * nscale_sha, 0.0)
        kp25_sun_v    = np.where(has_pai, kp25top    * nscale_sun, 0.0)
        kp25_sha_v    = np.where(has_pai, kp25top    * nscale_sha, 0.0)

        # Layer weighted mean — Fortran lines 150-154 (vectorized)
        vcmax25_profile_v = vcmax25_sun_v * fs_v + vcmax25_sha_v * (1.0 - fs_v)
        jmax25_profile_v  = jmax25_sun_v  * fs_v + jmax25_sha_v  * (1.0 - fs_v)
        rd25_profile_v    = rd25_sun_v    * fs_v + rd25_sha_v    * (1.0 - fs_v)
        kp25_profile_v    = kp25_sun_v    * fs_v + kp25_sha_v    * (1.0 - fs_v)

        # Reorder from top→bottom (ics order) to ic=1..ncan order
        # ics = [ncan, ncan-1, ..., 1], so reverse gives [ic=1, ic=2, ..., ic=ncan]
        vcmax25_sun_ord = vcmax25_sun_v[::-1]
        vcmax25_sha_ord = vcmax25_sha_v[::-1]
        jmax25_sun_ord  = jmax25_sun_v[::-1]
        jmax25_sha_ord  = jmax25_sha_v[::-1]
        rd25_sun_ord    = rd25_sun_v[::-1]
        rd25_sha_ord    = rd25_sha_v[::-1]
        kp25_sun_ord    = kp25_sun_v[::-1]
        kp25_sha_ord    = kp25_sha_v[::-1]
        vcmax25_prof_ord = vcmax25_profile_v[::-1]
        jmax25_prof_ord  = jmax25_profile_v[::-1]
        rd25_prof_ord    = rd25_profile_v[::-1]
        kp25_prof_ord    = kp25_profile_v[::-1]

        # Bulk JAX write-back — 12 transfers instead of ~300 per-element syncs
        _sl = slice(1, _ncan + 1)
        vcmax25_leaf    = vcmax25_leaf.at[p, _sl, isun].set(jnp.array(vcmax25_sun_ord))
        vcmax25_leaf    = vcmax25_leaf.at[p, _sl, isha].set(jnp.array(vcmax25_sha_ord))
        jmax25_leaf     = jmax25_leaf.at[p, _sl, isun].set(jnp.array(jmax25_sun_ord))
        jmax25_leaf     = jmax25_leaf.at[p, _sl, isha].set(jnp.array(jmax25_sha_ord))
        rd25_leaf       = rd25_leaf.at[p, _sl, isun].set(jnp.array(rd25_sun_ord))
        rd25_leaf       = rd25_leaf.at[p, _sl, isha].set(jnp.array(rd25_sha_ord))
        kp25_leaf       = kp25_leaf.at[p, _sl, isun].set(jnp.array(kp25_sun_ord))
        kp25_leaf       = kp25_leaf.at[p, _sl, isha].set(jnp.array(kp25_sha_ord))
        vcmax25_profile = vcmax25_profile.at[p, _sl].set(jnp.array(vcmax25_prof_ord))
        jmax25_profile  = jmax25_profile.at[p, _sl].set(jnp.array(jmax25_prof_ord))
        rd25_profile    = rd25_profile.at[p, _sl].set(jnp.array(rd25_prof_ord))
        kp25_profile    = kp25_profile.at[p, _sl].set(jnp.array(kp25_prof_ord))

        # Conservation check — Fortran lines 156-160 (numpy, no extra JAX syncs)
        numerical  = float(np.sum(vcmax25_prof_ord * _dpai[1:_ncan + 1]))
        analytical = vcmax25top * (1.0 - math.exp(-kn * (lai_p + sai_p))) / kn
        if abs(numerical - analytical) > 1.0e-6:
            endrun(msg='ERROR: CanopyNitrogenProfile: canopy integration error')

    return mlcanopy_inst._replace(
        vcmax25_leaf    = vcmax25_leaf,
        jmax25_leaf     = jmax25_leaf,
        rd25_leaf       = rd25_leaf,
        kp25_leaf       = kp25_leaf,
        vcmax25_profile = vcmax25_profile,
        jmax25_profile  = jmax25_profile,
        rd25_profile    = rd25_profile,
        kp25_profile    = kp25_profile,
    )