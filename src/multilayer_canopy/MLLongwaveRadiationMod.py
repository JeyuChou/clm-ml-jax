"""
JAX translation of MLLongwaveRadiationMod Fortran module.

Longwave radiation transfer through the multilayer canopy.
Provides one public driver (:func:`LongwaveRadiation`) and one
private solver (:func:`_Norman`).

Original Fortran module: MLLongwaveRadiationMod
Fortran lines 1-195
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import jax.numpy as jnp


def _tridiag_py(a, b, c, r, n):
    """Thomas algorithm using pure Python lists — no JAX, no numpy."""
    gam = [0.0] * (n + 1)
    u   = [0.0] * (n + 1)
    bet = b[1]
    u[1] = r[1] / bet
    for j in range(2, n + 1):
        gam[j] = c[j - 1] / bet
        bet     = b[j] - a[j] * gam[j]
        u[j]    = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 1, 0, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u

from clm_src_main.abortutils import endrun                           # noqa: F401
from clm_src_main.clm_varctl import iulog                           # noqa: F401
from clm_src_main.decompMod import bounds_type                      # noqa: F401
from clm_src_main.clm_varcon import sb                              # noqa: F401
from clm_src_main.PatchType import patch                            # noqa: F401
from multilayer_canopy.MLclm_varcon import emg                           # noqa: F401
from multilayer_canopy.MLclm_varctl import longwave_type                 # noqa: F401
from multilayer_canopy.MLclm_varpar import isun, isha, nlevmlcan         # noqa: F401
from multilayer_canopy.MLMathToolsMod import tridiag                     # noqa: F401
from multilayer_canopy.MLpftconMod import MLpftcon                       # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type           # noqa: F401


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def LongwaveRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Longwave radiation transfer through the multilayer canopy.

    Mirrors Fortran subroutine ``LongwaveRadiation`` (lines 30-45).

    Dispatches to :func:`_Norman` when ``longwave_type == 1``; any
    other value is fatal.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Multilayer canopy container; longwave flux
            fields are updated by the solver.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if longwave_type == 1:
        return _Norman(bounds, num_filter, filter_patch, mlcanopy_inst)
    else:
        endrun(msg=' ERROR: LongwaveRadiation: longwave_type not valid')
        return mlcanopy_inst    # Unreachable; satisfies type checker


# ---------------------------------------------------------------------------
# Private: Norman (1979) longwave radiative transfer
# ---------------------------------------------------------------------------

def _Norman(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Norman (1979) tridiagonal longwave radiative transfer.

    Mirrors Fortran subroutine ``Norman`` (private, lines 47-195).

    Reference: Bonan (2019) *Climate Change and Terrestrial Ecosystem
    Modeling*, Chapter 14.

    The algorithm mirrors the solar :func:`MLSolarRadiationMod._Norman`
    structure but operates on a single spectral band (no waveband loop)
    and includes leaf thermal emission as an additional source term.

    **Optical properties** (Fortran lines 97-101):

    .. code-block:: none

        omega = 1 - emleaf(pft)    [scattering coefficient]
        rho   = omega              [intercepted radiation fully reflected]
        tau   = 0                  [no transmission of intercepted radiation]

    **Layer emission** (Fortran lines 103-107):

    .. code-block:: none

        lw_source(ic) = [emleaf*sb*T_sun^4 * fracsun
                         + emleaf*sb*T_sha^4 * (1-fracsun)]
                        * (1 - td(ic))

    **Tridiagonal system** (Fortran lines 109-158): same equation
    ordering as the solar Norman solver — soil upward (m=1), soil
    downward (m=2), then for ``ic = nbot, ..., ntop-1`` upward (odd)
    and downward (even), finally top-layer upward and downward — with
    leaf emission entering the right-hand-side ``dtri`` terms.

    **Leaf absorption** (Fortran lines 170-181): absorbed flux per
    unit leaf area is the same for sunlit and shaded fractions:

    .. code-block:: none

        lwabs = emleaf * (lwdwn(ic) + lwupw(icm1)) * (1-td(ic))
                - 2 * lw_source(ic)
        lwleaf(ic, isun) = lwleaf(ic, isha) = lwabs / dpai(ic)

    Energy conservation is checked to within 1e-3 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; the following output fields
            are updated: ``lwup_canopy``, ``lwveg_canopy``,
            ``lwsoi_soil``, ``lwleaf_leaf``, ``lwupw_profile``,
            ``lwdwn_profile``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    neq: int = (nlevmlcan + 1) * 2    # Fortran: parameter neq = (nlevmlcan+1)*2

    emleaf = MLpftcon.emleaf

    lwup   = mlcanopy_inst.lwup_canopy
    lwveg  = mlcanopy_inst.lwveg_canopy
    lwsoi  = mlcanopy_inst.lwsoi_soil
    lwleaf = mlcanopy_inst.lwleaf_leaf
    lwupw  = mlcanopy_inst.lwupw_profile
    lwdwn  = mlcanopy_inst.lwdwn_profile

    for fp in range(1, num_filter + 1):            # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])
        _ncan = int(mlcanopy_inst.ncan_canopy[p])
        _ntop = int(mlcanopy_inst.ntop_canopy[p])
        _nbot = int(mlcanopy_inst.nbot_canopy[p])

        em_leaf = float(emleaf[pft])
        rho     = 1.0 - em_leaf
        tau     = 0.0                              # Fortran: tau = 0._r8

        # --- Pre-extract input slices as numpy (one JAX sync each) ---
        _tleaf_sun = np.asarray(mlcanopy_inst.tleaf_leaf[p, :, isun])
        _tleaf_sha = np.asarray(mlcanopy_inst.tleaf_leaf[p, :, isha])
        _fracsun   = np.asarray(mlcanopy_inst.fracsun_profile[p])
        _td        = np.asarray(mlcanopy_inst.td_profile[p])
        _dpai      = np.asarray(mlcanopy_inst.dpai_profile[p])
        _lwsky_p   = float(mlcanopy_inst.lwsky_forcing[p])
        _tg_p      = float(mlcanopy_inst.tg_soil[p])

        # Layer emission — vectorised numpy — Fortran lines 83-88
        ics = np.arange(_nbot, _ntop + 1)
        lw_source = np.zeros(nlevmlcan + 2)
        lw_sun = em_leaf * sb * _tleaf_sun[ics] ** 4
        lw_sha = em_leaf * sb * _tleaf_sha[ics] ** 4
        fs_arr = _fracsun[ics]
        lw_source[ics] = (lw_sun * fs_arr + lw_sha * (1.0 - fs_arr)) * (1.0 - _td[ics])

        # ------------------------------------------------------------------
        # Build tridiagonal system — Fortran lines 90-149 (pure Python lists)
        # ------------------------------------------------------------------
        atri = [0.0] * (neq + 1)
        btri = [0.0] * (neq + 1)
        ctri = [0.0] * (neq + 1)
        dtri = [0.0] * (neq + 1)

        m = 0

        # Soil: upward flux — Fortran lines 94-98
        m += 1
        atri[m] = 0.0
        btri[m] = 1.0
        ctri[m] = -(1.0 - emg)
        dtri[m] = emg * sb * _tg_p ** 4

        # Soil: downward flux — Fortran lines 100-108
        td_nb  = float(_td[_nbot])
        refld  = (1.0 - td_nb) * rho
        trand  = (1.0 - td_nb) * tau + td_nb
        aic    = refld - trand * trand / refld
        bic    = trand / refld
        m += 1
        atri[m] = -aic
        btri[m] = 1.0
        ctri[m] = -bic
        dtri[m] = (1.0 - bic) * lw_source[_nbot]

        # Leaf layers except top — Fortran lines 110-130
        for ic in range(_nbot, _ntop):             # Fortran: do ic = nbot, ntop-1
            td_ic  = float(_td[ic])
            refld  = (1.0 - td_ic) * rho
            trand  = (1.0 - td_ic) * tau + td_ic
            fic    = refld - trand * trand / refld
            eic    = trand / refld
            m += 1
            atri[m] = -eic;  btri[m] = 1.0
            ctri[m] = -fic;  dtri[m] = (1.0 - eic) * lw_source[ic]

            ic1    = ic + 1
            td_ic1 = float(_td[ic1])
            refld  = (1.0 - td_ic1) * rho
            trand  = (1.0 - td_ic1) * tau + td_ic1
            aic    = refld - trand * trand / refld
            bic    = trand / refld
            m += 1
            atri[m] = -aic;  btri[m] = 1.0
            ctri[m] = -bic;  dtri[m] = (1.0 - bic) * lw_source[ic1]

        # Top layer: upward flux — Fortran lines 132-140
        ic     = _ntop
        td_ic  = float(_td[ic])
        refld  = (1.0 - td_ic) * rho
        trand  = (1.0 - td_ic) * tau + td_ic
        fic    = refld - trand * trand / refld
        eic    = trand / refld
        m += 1
        atri[m] = -eic;  btri[m] = 1.0
        ctri[m] = -fic;  dtri[m] = (1.0 - eic) * lw_source[ic]

        # Top layer: downward flux — Fortran lines 142-146
        m += 1
        atri[m] = 0.0;  btri[m] = 1.0;  ctri[m] = 0.0;  dtri[m] = _lwsky_p

        # Solve — pure Python Thomas algorithm (no JAX) — Fortran line 148
        utri = _tridiag_py(atri, btri, ctri, dtri, m)

        # ------------------------------------------------------------------
        # Unpack solution into numpy arrays — Fortran lines 154-163
        # ------------------------------------------------------------------
        lwupw_new = np.zeros(nlevmlcan + 2)
        lwdwn_new = np.zeros(nlevmlcan + 2)
        k = 0
        k += 1;  lwupw_new[0] = utri[k]
        k += 1;  lwdwn_new[0] = utri[k]
        for ic in range(_nbot, _ntop + 1):
            k += 1;  lwupw_new[ic] = utri[k]
            k += 1;  lwdwn_new[ic] = utri[k]

        # ------------------------------------------------------------------
        # Ground absorption — Fortran line 165
        # ------------------------------------------------------------------
        _lwsoi_p = lwdwn_new[0] - lwupw_new[0]

        # ------------------------------------------------------------------
        # Leaf layer absorption — vectorised numpy — Fortran lines 167-181
        # ------------------------------------------------------------------
        icm1s    = np.where(ics == _nbot, 0, ics - 1)
        lwabs_arr = (em_leaf
                     * (lwdwn_new[ics] + lwupw_new[icm1s])
                     * (1.0 - _td[ics])
                     - 2.0 * lw_source[ics])
        dpai_ics = _dpai[ics]
        lwleaf_arr = lwabs_arr / dpai_ics
        _lwveg_p = float(np.sum(lwabs_arr))

        # Local numpy arrays for lwleaf (same value for isun and isha)
        lwleaf_new = np.zeros(nlevmlcan + 2)
        lwleaf_new[ics] = lwleaf_arr

        # Canopy upward longwave — Fortran line 183
        _lwup_p = lwupw_new[_ntop]

        # Conservation check — Fortran lines 185-189
        err = (_lwsky_p - _lwup_p) - (_lwveg_p + _lwsoi_p)
        if abs(err) > 1.0e-3:
            endrun(msg='ERROR: Norman: total longwave conservation error')

        # DEBUG: print LW diagnostics for first few calls
        # ------------------------------------------------------------------
        # Batch write-back (one _replace per patch)
        # ------------------------------------------------------------------
        _sl = slice(0, _ntop + 1)
        lwup  = lwup.at[p].set(_lwup_p)
        lwveg = lwveg.at[p].set(_lwveg_p)
        lwsoi = lwsoi.at[p].set(_lwsoi_p)
        lwleaf = (lwleaf
                  .at[p, _sl, isun].set(jnp.array(lwleaf_new[_sl]))
                  .at[p, _sl, isha].set(jnp.array(lwleaf_new[_sl])))
        lwupw = lwupw.at[p, _sl].set(jnp.array(lwupw_new[_sl]))
        lwdwn = lwdwn.at[p, _sl].set(jnp.array(lwdwn_new[_sl]))

    return mlcanopy_inst._replace(
        lwup_canopy   = lwup,
        lwveg_canopy  = lwveg,
        lwsoi_soil    = lwsoi,
        lwleaf_leaf   = lwleaf,
        lwupw_profile = lwupw,
        lwdwn_profile = lwdwn,
    )