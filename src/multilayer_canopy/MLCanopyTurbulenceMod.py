"""
JAX translation of MLCanopyTurbulenceMod Fortran module.

Canopy turbulence parameterisation: roughness sublayer (RSL) theory
of Harman and Finnigan (2008), wind speed profile, and aerodynamic
conductances.

Public routines
---------------
- :func:`CanopyTurbulence`: main driver.
- :func:`LookupPsihatINI`: initialise RSL psihat look-up tables from
  a NetCDF file.

Private helpers
---------------
- :func:`_HF2008`: HF2008 RSL canopy turbulence.
- :func:`_GetObu`: solve for Obukhov length.
- :func:`_ObuFunc`: ObuFunc callback for hybrid root solver.
- :func:`_GetBeta`: β = u*/u(h) for a given stability.
- :func:`_GetPrSc`: Prandtl/Schmidt number at canopy top.
- :func:`_GetPsiRSL`: RSL-modified stability functions.
- :func:`_phim_monin_obukhov`: MO φ for momentum.
- :func:`_phic_monin_obukhov`: MO φ for scalars.
- :func:`_psim_monin_obukhov`: MO ψ for momentum.
- :func:`_psic_monin_obukhov`: MO ψ for scalars.
- :func:`_LookupPsihat`: bilinear interpolation in psihat table.
- :func:`_RoughnessLength`: roughness length for momentum.
- :func:`_WindProfile`: wind speed above and within canopy.
- :func:`_AerodynamicConductance`: aerodynamic conductances.

Original Fortran module: MLCanopyTurbulenceMod
Fortran lines 1-560
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from clm_src_main.abortutils import endrun                           # noqa: F401
from clm_src_main.clm_varctl import iulog, rslfile                  # noqa: F401
from clm_src_main.clm_varcon import grav, vkc, pi as rpi            # noqa: F401
from multilayer_canopy.MLclm_varcon import (                             # noqa: F401
    mmh2o, mmdry, cd, eta_max,
    beta_neutral_max, cr, z0mg, LcL_min, LcL_max, aH12,
    c2, dtLgridM, zdtgridM, psigridM,
    dtLgridH, zdtgridH, psigridH,
    nZ, nL,
    Pr0, Pr1, Pr2,
    z0mg, ra_max,
)
from multilayer_canopy.MLclm_varctl import (                             # noqa: F401
    turb_type, sparse_canopy_type, HF_extension_type,
)
from multilayer_canopy.MLMathToolsMod import hybrid, hybrid_scalar        # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type           # noqa: F401

# Python-float aliases for aH12 elements — used inside _ObuFuncPure which is called
# on every solver iteration; accessing aH12[i] (a JAX array) would force an XLA sync
# each time, adding ~100–200 syncs per _GetObu call.
_aH12_0: float = 0.89
_aH12_1: float = -0.07
_aH12_2: float = 2.19

# ---------------------------------------------------------------------------
# Cached 1D views of psihat look-up table grids — populated by LookupPsihatINI.
# _LookupPsihat creates a 276-element negated array (-zdtg) on every call for
# np.searchsorted; with ~220 calls/patch/step that's ~110,000 allocations/run.
# Using module-level pre-computed arrays eliminates those allocations and the
# 2D→1D slice operations (dtLgrid[0], zdtgrid[:, 0]) done on every call.
# ---------------------------------------------------------------------------
_zdtgM_1d:     np.ndarray = np.empty(0)   # zdtgridM[:, 0]  (populated after init)
_dtLgM_1d:     np.ndarray = np.empty(0)   # dtLgridM[0]
_neg_zdtgM_1d: np.ndarray = np.empty(0)   # -zdtgridM[:, 0]
_nZ_M: int = nZ
_nL_M: int = nL
_zdtgH_1d:     np.ndarray = np.empty(0)   # zdtgridH[:, 0]
_dtLgH_1d:     np.ndarray = np.empty(0)   # dtLgridH[0]
_neg_zdtgH_1d: np.ndarray = np.empty(0)   # -zdtgridH[:, 0]
_nZ_H: int = nZ
_nL_H: int = nL

# ===========================================================================
# Private: Monin-Obukhov stability functions
# ===========================================================================

def _phim_monin_obukhov(zeta: float) -> float:
    """
    Monin-Obukhov φ stability function for momentum.

    Mirrors Fortran function ``phim_monin_obukhov`` (lines 330-343).

    Reference: Bonan et al. (2018), eq. (A10).

    .. code-block:: none

        zeta < 0 (unstable): phi = 1 / (1 - 16*zeta)^(1/4)
        zeta ≥ 0 (stable):   phi = 1 + 5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter.

    Returns:
        φ for momentum.
    """
    if zeta < 0.0:
        return 1.0 / math.sqrt(math.sqrt(1.0 - 16.0 * zeta))
    else:
        return 1.0 + 5.0 * zeta


def _phic_monin_obukhov(zeta: float) -> float:
    """
    Monin-Obukhov φ stability function for scalars.

    Mirrors Fortran function ``phic_monin_obukhov`` (lines 345-358).

    Reference: Bonan et al. (2018), eq. (A11).

    .. code-block:: none

        zeta < 0 (unstable): phi = 1 / (1 - 16*zeta)^(1/2)
        zeta ≥ 0 (stable):   phi = 1 + 5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter.

    Returns:
        φ for scalars.
    """
    if zeta < 0.0:
        return 1.0 / math.sqrt(1.0 - 16.0 * zeta)
    else:
        return 1.0 + 5.0 * zeta


def _psim_monin_obukhov(zeta: float) -> float:
    """
    Monin-Obukhov ψ stability function for momentum.

    Mirrors Fortran function ``psim_monin_obukhov`` (lines 360-378).

    Reference: Bonan et al. (2018), eq. (A12).

    .. code-block:: none

        zeta < 0 (unstable):
            x   = (1 - 16*zeta)^(1/4)
            psi = 2*log((1+x)/2) + log((1+x^2)/2) - 2*atan(x) + pi/2
        zeta ≥ 0 (stable):
            psi = -5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter.

    Returns:
        ψ for momentum.
    """
    if zeta < 0.0:
        x = math.sqrt(math.sqrt(1.0 - 16.0 * zeta))
        return (2.0 * math.log((1.0 + x) / 2.0)
                + math.log((1.0 + x * x) / 2.0)
                - 2.0 * math.atan(x)
                + rpi * 0.5)
    else:
        return -5.0 * zeta


def _psic_monin_obukhov(zeta: float) -> float:
    """
    Monin-Obukhov ψ stability function for scalars.

    Mirrors Fortran function ``psic_monin_obukhov`` (lines 380-395).

    Reference: Bonan et al. (2018), eq. (A13).

    .. code-block:: none

        zeta < 0 (unstable):
            x   = (1 - 16*zeta)^(1/4)
            psi = 2*log((1+x^2)/2)
        zeta ≥ 0 (stable):
            psi = -5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter.

    Returns:
        ψ for scalars.
    """
    if zeta < 0.0:
        x = math.sqrt(math.sqrt(1.0 - 16.0 * zeta))
        return 2.0 * math.log((1.0 + x * x) / 2.0)
    else:
        return -5.0 * zeta


# ===========================================================================
# Private: RSL psihat look-up table interpolation
# ===========================================================================

def _LookupPsihat(
    zdt: float,
    dtL: float,
    zdtgrid,    # shape (nZ, 1)
    dtLgrid,    # shape (1, nL)
    psigrid,    # shape (nZ, nL)
) -> float:
    """
    Bilinear interpolation of the RSL psihat function from a look-up table.

    Mirrors Fortran subroutine ``LookupPsihat`` (lines 398-448).

    Reference: Bonan et al. (2018), eq. (A21).

    The returned value is the unscaled amplitude ``A[(z-h)/(h-d),(h-d)/L]``;
    it must be multiplied by ``c1`` before it fully represents psihat.

    Look-up table grid conventions (1-based in Fortran, 0-based here):

    - ``zdtgrid`` decreases with increasing row index (height above canopy
      decreasing toward zero as index increases); boundary clamping applies
      when ``zdt`` exceeds the table range.
    - ``dtLgrid`` increases with increasing column index; boundary clamping
      applies when ``dtL`` is outside the table range.

    Args:
        zdt: Normalised height above canopy ``(z-h)/(h-d)``.
        dtL: Stability ratio ``(h-d)/L``.
        zdtgrid: Grid of ``zdt`` values; shape ``(nZ, 1)``.
        dtLgrid: Grid of ``dtL`` values; shape ``(1, nL)``.
        psigrid: Psihat values on the grid; shape ``(nZ, nL)``.

    Returns:
        Interpolated (unscaled) psihat value.
    """
    dtLg = dtLgrid[0]       # 1-D view, length nL (no copy)
    zdtg = zdtgrid[:, 0]    # 1-D view, length nZ (no copy); DECREASING
    nZ_tbl = len(zdtg)
    nL_tbl = len(dtLg)

    # --- dtL bracketing — Fortran lines 412-428 ---
    # dtLg is increasing → np.searchsorted gives the insertion point
    if dtL <= dtLg[0]:
        L1 = 0; L2 = 0; wL1 = 0.5; wL2 = 0.5
    elif dtL > dtLg[-1]:
        L1 = nL_tbl - 1; L2 = nL_tbl - 1; wL1 = 0.5; wL2 = 0.5
    else:
        # Clamp to nL_tbl-1 in case dtL == dtLg[-1] exactly
        L2 = min(int(np.searchsorted(dtLg, dtL, side='right')), nL_tbl - 1)
        L1 = L2 - 1
        wL1 = (dtLg[L2] - dtL) / (dtLg[L2] - dtLg[L1])
        wL2 = 1.0 - wL1

    # --- zdt bracketing — Fortran lines 430-448 (zdtg DECREASES) ---
    # Negate both to apply searchsorted on the increasing negative grid
    if zdt > zdtg[0]:
        Z1 = 0; Z2 = 0; wZ1 = 0.5; wZ2 = 0.5
    elif zdt < zdtg[-1]:
        Z1 = nZ_tbl - 1; Z2 = nZ_tbl - 1; wZ1 = 0.5; wZ2 = 0.5
    else:
        # Clamp to nZ_tbl-2 in case zdt == zdtg[-1] exactly
        Z1 = min(int(np.searchsorted(-zdtg, -zdt, side='right')) - 1, nZ_tbl - 2)
        Z2 = Z1 + 1
        wZ1 = (zdt - zdtg[Z2]) / (zdtg[Z1] - zdtg[Z2])
        wZ2 = 1.0 - wZ1

    # Bilinear interpolation — Fortran lines 450-452
    return (wZ1 * wL1 * psigrid[Z1, L1]
            + wZ2 * wL1 * psigrid[Z2, L1]
            + wZ1 * wL2 * psigrid[Z1, L2]
            + wZ2 * wL2 * psigrid[Z2, L2])


# ---------------------------------------------------------------------------
# Fast specialized lookups using module-level pre-computed 1D arrays.
# These eliminate the per-call slice and negation overhead of _LookupPsihat.
# Called by _GetPsiRSL after LookupPsihatINI has populated the caches.
# ---------------------------------------------------------------------------

def _LookupPsihatM(zdt: float, dtL: float) -> float:
    """Momentum psihat lookup using module-level cached 1D arrays."""
    if dtL <= _dtLgM_1d[0]:
        L1 = 0; L2 = 0; wL1 = 0.5; wL2 = 0.5
    elif dtL > _dtLgM_1d[-1]:
        L1 = _nL_M - 1; L2 = _nL_M - 1; wL1 = 0.5; wL2 = 0.5
    else:
        L2 = min(int(np.searchsorted(_dtLgM_1d, dtL, side='right')), _nL_M - 1)
        L1 = L2 - 1
        wL1 = (_dtLgM_1d[L2] - dtL) / (_dtLgM_1d[L2] - _dtLgM_1d[L1])
        wL2 = 1.0 - wL1
    if zdt > _zdtgM_1d[0]:
        Z1 = 0; Z2 = 0; wZ1 = 0.5; wZ2 = 0.5
    elif zdt < _zdtgM_1d[-1]:
        Z1 = _nZ_M - 1; Z2 = _nZ_M - 1; wZ1 = 0.5; wZ2 = 0.5
    else:
        Z1 = min(int(np.searchsorted(_neg_zdtgM_1d, -zdt, side='right')) - 1, _nZ_M - 2)
        Z2 = Z1 + 1
        wZ1 = (zdt - _zdtgM_1d[Z2]) / (_zdtgM_1d[Z1] - _zdtgM_1d[Z2])
        wZ2 = 1.0 - wZ1
    return (wZ1 * wL1 * psigridM[Z1, L1]
            + wZ2 * wL1 * psigridM[Z2, L1]
            + wZ1 * wL2 * psigridM[Z1, L2]
            + wZ2 * wL2 * psigridM[Z2, L2])


def _LookupPsihatH(zdt: float, dtL: float) -> float:
    """Heat/scalar psihat lookup using module-level cached 1D arrays."""
    if dtL <= _dtLgH_1d[0]:
        L1 = 0; L2 = 0; wL1 = 0.5; wL2 = 0.5
    elif dtL > _dtLgH_1d[-1]:
        L1 = _nL_H - 1; L2 = _nL_H - 1; wL1 = 0.5; wL2 = 0.5
    else:
        L2 = min(int(np.searchsorted(_dtLgH_1d, dtL, side='right')), _nL_H - 1)
        L1 = L2 - 1
        wL1 = (_dtLgH_1d[L2] - dtL) / (_dtLgH_1d[L2] - _dtLgH_1d[L1])
        wL2 = 1.0 - wL1
    if zdt > _zdtgH_1d[0]:
        Z1 = 0; Z2 = 0; wZ1 = 0.5; wZ2 = 0.5
    elif zdt < _zdtgH_1d[-1]:
        Z1 = _nZ_H - 1; Z2 = _nZ_H - 1; wZ1 = 0.5; wZ2 = 0.5
    else:
        Z1 = min(int(np.searchsorted(_neg_zdtgH_1d, -zdt, side='right')) - 1, _nZ_H - 2)
        Z2 = Z1 + 1
        wZ1 = (zdt - _zdtgH_1d[Z2]) / (_zdtgH_1d[Z1] - _zdtgH_1d[Z2])
        wZ2 = 1.0 - wZ1
    return (wZ1 * wL1 * psigridH[Z1, L1]
            + wZ2 * wL1 * psigridH[Z2, L1]
            + wZ1 * wL2 * psigridH[Z1, L2]
            + wZ2 * wL2 * psigridH[Z2, L2])


# ===========================================================================
# Private: beta = u*/u(h)
# ===========================================================================

def _GetBeta(beta_neutral: float, LcL: float) -> float:
    """
    Calculate β = u*/u(h) for the current Obukhov length stability.

    Mirrors Fortran subroutine ``GetBeta`` (lines 245-290).

    Reference: Bonan et al. (2018), eqs. (A22)-(A24).

    **Unstable** (``LcL ≤ 0``): solves the quadratic equation for β²:

    .. code-block:: none

        β² = (-b + sqrt(b²-4ac)) / 2a
        where a=1, b=16*LcL*β_n^4, c=-β_n^4

    **Stable** (``LcL > 0``): solves the depressed cubic equation for β
    via Cardano's formula:

    .. code-block:: none

        5*LcL*β^3 + β - β_n = 0

    Error check: ``|β*φm(LcL*β²) - β_n| < 1e-6``.

    Args:
        beta_neutral: Neutral value of β = u*/u(h).
        LcL: Canopy density scale Lc divided by Obukhov length L.

    Returns:
        β value.
    """
    if LcL <= 0.0:                                 # Fortran lines 258-261: unstable quadratic
        aa = 1.0
        bb = 16.0 * LcL * beta_neutral ** 4
        cc = -(beta_neutral ** 4)
        beta = math.sqrt((-bb + math.sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa))
    else:                                          # Fortran lines 263-272: stable cubic
        aa = 5.0 * LcL
        bb = 0.0
        cc = 1.0
        dd = -beta_neutral
        qq = ((2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * aa**2 * dd)**2
              - 4.0 * (bb**2 - 3.0 * aa * cc)**3)
        qq = math.sqrt(qq)
        rr = 0.5 * (qq + 2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * aa**2 * dd)
        rr = rr ** (1.0 / 3.0)
        beta = -(bb + rr) / (3.0 * aa) - (bb**2 - 3.0 * aa * cc) / (3.0 * aa * rr)

    # Error check — Fortran lines 274-278
    y   = LcL * beta ** 2
    fy  = _phim_monin_obukhov(y)
    err = beta * fy - beta_neutral
    if abs(err) > 1.0e-6:
        endrun(msg=' ERROR: GetBeta: beta error')

    return beta


# ===========================================================================
# Private: Prandtl / Schmidt number
# ===========================================================================

def _GetPrSc(
    beta_neutral: float,
    beta_neutral_max: float,
    LcL: float,
) -> float:
    """
    Calculate the Prandtl (= Schmidt) number at the canopy top.

    Mirrors Fortran subroutine ``GetPrSc`` (lines 292-310).

    Reference: Bonan et al. (2018), eqs. (A25), (A34).

    .. code-block:: none

        PrSc = Pr0 + Pr1 * tanh(Pr2 * LcL)

    For sparse canopies (``sparse_canopy_type == 1``):

    .. code-block:: none

        PrSc = (1 - beta_n/beta_n_max) * 1.0
               + (beta_n/beta_n_max) * PrSc

    Args:
        beta_neutral: Neutral β.
        beta_neutral_max: Maximum allowed neutral β.
        LcL: Canopy density scale / Obukhov length.

    Returns:
        PrSc value.
    """
    PrSc = Pr0 + Pr1 * math.tanh(Pr2 * LcL)       # Fortran line 304

    if sparse_canopy_type == 1:                    # Fortran lines 306-308
        f = beta_neutral / beta_neutral_max
        PrSc = (1.0 - f) * 1.0 + f * PrSc

    return PrSc


# ===========================================================================
# Private: RSL-modified psi functions
# ===========================================================================

def _GetPsiRSL(
    za: float,
    hc: float,
    disp: float,
    obu: float,
    beta: float,
    PrSc: float,
) -> Tuple[float, float, float, float]:
    """
    RSL-modified ψ stability functions for momentum and scalars.

    Mirrors Fortran subroutine ``GetPsiRSL`` (lines 312-398).

    Reference: Bonan et al. (2018), appendix A2.

    Computes the integrated stability corrections between ``hc`` and
    ``za`` including the roughness sublayer psihat modifications.

    **Momentum** (Bonan et al. 2018, eq. A16, A21, 19):

    .. code-block:: none

        phim = phim_monin_obukhov((hc-disp)/obu)
        c1   = (1 - vkc/(2*beta*phim)) * exp(0.5*c2)
        psim = -psim1 + psim2 + c1*psihat_m(za) - c1*psihat_m(hc) + vkc/beta

    **Scalars** (eqs. A19, A21, 20):

    .. code-block:: none

        phic = phic_monin_obukhov((hc-disp)/obu)
        c1   = (1 - PrSc*vkc/(2*beta*phic)) * exp(0.5*c2)
        psic = -psic1 + psic2 + c1*psihat_h(za) - c1*psihat_h(hc)

    Args:
        za: Atmospheric height (m).
        hc: Canopy top height (m).
        disp: Displacement height (m).
        obu: Obukhov length (m).
        beta: u*/u(h) at canopy top.
        PrSc: Prandtl/Schmidt number.

    Returns:
        Tuple ``(psim, psic, psim2, psim_hat2)`` where:
        - ``psim``: ψ for momentum including RSL (evaluated za→hc).
        - ``psic``: ψ for scalars including RSL.
        - ``psim2``: MO ψ for momentum evaluated at hc.
        - ``psim_hat2``: RSL psihat for momentum evaluated at hc.
    """
    dt = hc - disp                                 # Fortran line 358
    zdt_za  = (za - hc) / dt                       # normalised height at za
    zdt_hc  = 0.0                                  # (hc - hc) / dt == 0 always
    dtL     = dt / obu                             # stability ratio

    # Momentum — Fortran lines 360-376
    phim = _phim_monin_obukhov((hc - disp) / obu)
    c1_m = (1.0 - vkc / (2.0 * beta * phim)) * math.exp(0.5 * c2)

    psim_hat1_raw = _LookupPsihatM(zdt_za, dtL)
    psim_hat2_raw = _LookupPsihatM(zdt_hc, dtL)
    psim_hat1 = psim_hat1_raw * c1_m
    psim_hat2 = psim_hat2_raw * c1_m

    psim1 = _psim_monin_obukhov((za  - disp) / obu)
    psim2 = _psim_monin_obukhov((hc  - disp) / obu)

    psim = -psim1 + psim2 + psim_hat1 - psim_hat2 + vkc / beta

    # Scalars — Fortran lines 378-393
    phic = _phic_monin_obukhov((hc - disp) / obu)
    c1_c = (1.0 - PrSc * vkc / (2.0 * beta * phic)) * math.exp(0.5 * c2)

    psic_hat1_raw = _LookupPsihatH(zdt_za, dtL)
    psic_hat2_raw = _LookupPsihatH(zdt_hc, dtL)
    psic_hat1 = psic_hat1_raw * c1_c
    psic_hat2 = psic_hat2_raw * c1_c

    psic1 = _psic_monin_obukhov((za  - disp) / obu)
    psic2 = _psic_monin_obukhov((hc  - disp) / obu)

    psic = -psic1 + psic2 + psic_hat1 - psic_hat2

    return psim, psic, psim2, psim_hat2


# ===========================================================================
# Private: ObuFunc callback (hybrid root-finder signature)
# ===========================================================================

def _ObuFunc(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    obu_val: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Solve for the Obukhov length at a single patch.

    Mirrors Fortran subroutine ``ObuFunc`` (lines 178-244).

    Reference: Bonan et al. (2018), eqs. (A28)-(A33), (19)-(20).

    For a trial ``obu_val``:

    1. Clamp ``obu_val`` to avoid singularity near zero using
       ``LcL_min`` / ``LcL_max``.
    2. Compute neutral β and stability-corrected β via
       :func:`_GetBeta` (HF RSL and no-RSL versions), then blend.
    3. Compute displacement height ``zdisp`` (with optional sparse
       canopy correction).
    4. Compute PrSc via :func:`_GetPrSc`.
    5. Compute ψ functions via :func:`_GetPsiRSL`.
    6. Derive ``ustar``, ``tstar``, ``qstar``,
       ``gac_to_hc``, and the new Obukhov length.
    7. Return ``obu_dif = obu_new - obu_val``.

    Matches the :data:`MLMathToolsMod.FuncType` callback signature.

    Args:
        p: Patch index.
        ic: Unused (set to 0 by caller); required by callback contract.
        il: Unused (set to 0 by caller); required by callback contract.
        mlcanopy_inst: Canopy container; ``zdisp``, ``beta``, ``PrSc``,
            ``ustar``, ``gac_to_hc``, and ``obu`` are updated.
        obu_val: Trial Obukhov length (m).

    Returns:
        Tuple ``(obu_dif, mlcanopy_inst)``.
    """
    Lc_p   = float(mlcanopy_inst.Lc_canopy[p])
    ztop_p = float(mlcanopy_inst.ztop_canopy[p])
    lai_p  = float(mlcanopy_inst.lai_canopy[p])
    sai_p  = float(mlcanopy_inst.sai_canopy[p])
    zref_p = float(mlcanopy_inst.zref_forcing[p])
    uref_p = float(mlcanopy_inst.uref_forcing[p])
    thref_p  = float(mlcanopy_inst.thref_forcing[p])
    thvref_p = float(mlcanopy_inst.thvref_forcing[p])
    qref_p   = float(mlcanopy_inst.qref_forcing[p])
    rhomol_p = float(mlcanopy_inst.rhomol_forcing[p])
    taf_p    = float(mlcanopy_inst.taf_canopy[p])
    qaf_p    = float(mlcanopy_inst.qaf_canopy[p])

    # Clamp obu_val away from zero — Fortran lines 195-202
    obu_min_stable   = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    if obu_val >= 0.0:
        obu_cur = max(obu_val, obu_min_stable)
    else:
        obu_cur = min(obu_val, obu_max_unstable)
    LcL = Lc_p / obu_cur

    # Neutral beta — Fortran lines 204-206
    c1_n         = (vkc / math.log((ztop_p + z0mg) / z0mg)) ** 2
    beta_neutral = min(math.sqrt(c1_n + cr * (lai_p + sai_p)), beta_neutral_max)

    # Stability-corrected beta (HF + no-RSL blend) — Fortran lines 208-215
    beta_HF    = _GetBeta(beta_neutral, LcL)
    beta_norsl = _GetBeta(vkc / 2.0, LcL)
    if LcL > aH12[1]:                             # Fortran: aH12(2) → 0-based index 1
        beta_val = beta_HF
    else:
        beta_val = (beta_norsl
                    + (beta_HF - beta_norsl)
                    / (1.0 + aH12[0] * abs(LcL - aH12[1]) ** aH12[2]))

    # Displacement height — Fortran lines 217-224
    hc_minus_d = beta_val ** 2 * Lc_p
    if sparse_canopy_type == 1:
        hc_minus_d *= (1.0 - math.exp(-0.25 * (lai_p + sai_p) / beta_val ** 2))
    hc_minus_d = min(ztop_p, hc_minus_d)
    zdisp_val  = ztop_p - hc_minus_d

    if (zref_p - zdisp_val) < 0.0:
        endrun(msg=' ERROR: ObuFunc: zdisp height > zref')

    # PrSc — Fortran line 228
    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL)

    # psi functions — Fortran lines 230-233
    psim, psic, _dum1, _dum2 = _GetPsiRSL(
        zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val
    )

    # Friction velocity, temperature and humidity scales — Fortran lines 235-243
    zlog = math.log((zref_p - zdisp_val) / (ztop_p - zdisp_val))
    ustar_val   = uref_p * vkc / (zlog + psim)
    tstar       = (thref_p  - taf_p) * vkc / (zlog + psic)
    qstar       = (qref_p   - qaf_p) * vkc / (zlog + psic)
    gac_to_hc_v = rhomol_p * vkc * ustar_val / (zlog + psic)

    # New Obukhov length — Fortran lines 245-249
    tvstar  = tstar + 0.61 * thref_p * qstar
    obu_new = ustar_val ** 2 * thvref_p / (vkc * grav * tvstar)
    obu_dif = obu_new - obu_val

    mlcanopy_inst = mlcanopy_inst._replace(
        zdisp_canopy     = mlcanopy_inst.zdisp_canopy.at[p].set(zdisp_val),
        beta_canopy      = mlcanopy_inst.beta_canopy.at[p].set(beta_val),
        PrSc_canopy      = mlcanopy_inst.PrSc_canopy.at[p].set(PrSc_val),
        ustar_canopy     = mlcanopy_inst.ustar_canopy.at[p].set(ustar_val),
        gac_to_hc_canopy = mlcanopy_inst.gac_to_hc_canopy.at[p].set(gac_to_hc_v),
        obu_canopy       = mlcanopy_inst.obu_canopy.at[p].set(obu_cur),
    )
    return obu_dif, mlcanopy_inst


# ===========================================================================
# Private: ObuFuncPure — pure-scalar variant (no JAX reads/writes)
# ===========================================================================

def _ObuFuncPure(
    obu_val: float,
    *,
    Lc_p: float,
    ztop_p: float,
    lai_p: float,
    sai_p: float,
    zref_p: float,
    uref_p: float,
    thref_p: float,
    thvref_p: float,
    qref_p: float,
    rhomol_p: float,
    taf_p: float,
    qaf_p: float,
) -> float:
    """Pure-scalar version of _ObuFunc: no JAX reads/writes, returns obu_dif only."""
    obu_min_stable   = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    if obu_val >= 0.0:
        obu_cur = max(obu_val, obu_min_stable)
    else:
        obu_cur = min(obu_val, obu_max_unstable)
    LcL = Lc_p / obu_cur

    c1_n         = (vkc / math.log((ztop_p + z0mg) / z0mg)) ** 2
    beta_neutral = min(math.sqrt(c1_n + cr * (lai_p + sai_p)), beta_neutral_max)

    beta_HF    = _GetBeta(beta_neutral, LcL)
    beta_norsl = _GetBeta(vkc / 2.0, LcL)
    if LcL > _aH12_1:
        beta_val = beta_HF
    else:
        beta_val = (beta_norsl
                    + (beta_HF - beta_norsl)
                    / (1.0 + _aH12_0 * abs(LcL - _aH12_1) ** _aH12_2))

    hc_minus_d = beta_val ** 2 * Lc_p
    if sparse_canopy_type == 1:
        hc_minus_d *= (1.0 - math.exp(-0.25 * (lai_p + sai_p) / beta_val ** 2))
    hc_minus_d = min(ztop_p, hc_minus_d)
    zdisp_val  = ztop_p - hc_minus_d

    if (zref_p - zdisp_val) < 0.0:
        endrun(msg=' ERROR: ObuFunc: zdisp height > zref')

    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL)

    psim, psic, _dum1, _dum2 = _GetPsiRSL(
        zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val
    )

    zlog        = math.log((zref_p - zdisp_val) / (ztop_p - zdisp_val))
    ustar_val   = uref_p * vkc / (zlog + psim)
    tstar       = (thref_p  - taf_p) * vkc / (zlog + psic)
    qstar       = (qref_p   - qaf_p) * vkc / (zlog + psic)

    tvstar  = tstar + 0.61 * thref_p * qstar
    obu_new = ustar_val ** 2 * thvref_p / (vkc * grav * tvstar)
    obu_dif = obu_new - obu_val
    return obu_dif


# ===========================================================================
# Private: GetObu driver
# ===========================================================================

def _GetObu(p: int, mlcanopy_inst: mlcanopy_type) -> mlcanopy_type:
    """
    Solve for the Obukhov length using the hybrid root solver.

    Mirrors Fortran subroutine ``GetObu`` (lines 160-178).

    Passes :func:`_ObuFunc` to :func:`hybrid` with initial brackets
    ``obu0 = 100`` and ``obu1 = -100`` m, tolerance 0.1 m.  The
    return value from :func:`hybrid` is discarded; the converged
    ``obu`` is the value used inside :func:`_ObuFunc` to compute
    ``ustar`` and is stored in ``mlcanopy_inst.obu_canopy[p]``.

    Args:
        p: Patch index.
        mlcanopy_inst: Canopy container; Obukhov length and dependent
            quantities are updated via :func:`_ObuFunc`.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # Pre-extract all scalar inputs once (avoids JAX sync on every solver iteration)
    _kwargs = dict(
        Lc_p      = float(mlcanopy_inst.Lc_canopy[p]),
        ztop_p    = float(mlcanopy_inst.ztop_canopy[p]),
        lai_p     = float(mlcanopy_inst.lai_canopy[p]),
        sai_p     = float(mlcanopy_inst.sai_canopy[p]),
        zref_p    = float(mlcanopy_inst.zref_forcing[p]),
        uref_p    = float(mlcanopy_inst.uref_forcing[p]),
        thref_p   = float(mlcanopy_inst.thref_forcing[p]),
        thvref_p  = float(mlcanopy_inst.thvref_forcing[p]),
        qref_p    = float(mlcanopy_inst.qref_forcing[p]),
        rhomol_p  = float(mlcanopy_inst.rhomol_forcing[p]),
        taf_p     = float(mlcanopy_inst.taf_canopy[p]),
        qaf_p     = float(mlcanopy_inst.qaf_canopy[p]),
    )

    def _obu_closure(obu_val: float) -> float:
        return _ObuFuncPure(obu_val, **_kwargs)

    obu0: float = 100.0
    obu1: float = -100.0
    tol:  float = 0.1

    obu_converged = hybrid_scalar('GetObu', _obu_closure, obu0, obu1, tol)

    # Write-back: one call to _ObuFunc to update all 6 fields in mlcanopy_inst
    _dummy, mlcanopy_inst = _ObuFunc(p, 0, 0, mlcanopy_inst, obu_converged)
    return mlcanopy_inst


# ===========================================================================
# Private: RoughnessLength
# ===========================================================================

def _RoughnessLength(p: int, mlcanopy_inst: mlcanopy_type) -> mlcanopy_type:
    """
    Calculate the roughness length for momentum via inline bisection.

    Mirrors Fortran subroutine ``RoughnessLength`` (lines 454-520).

    Uses the self-consistent definition:

    .. code-block:: none

        z0m = (hc - d) * exp(-vkc/beta) * exp(-psim_hc + psim(z0m)) * exp(psim_hat_hc)

    where ``psim_hc`` and ``psim_hat_hc`` come from
    :func:`_GetPsiRSL` evaluated at the canopy top.
    Bisection between ``aval = ztop`` and ``bval = 0`` to tolerance
    0.001 m; maximum 20 iterations.

    Args:
        p: Patch index.
        mlcanopy_inst: Canopy container; ``z0m_canopy[p]`` is updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    ztop_p  = float(mlcanopy_inst.ztop_canopy[p])
    zdisp_p = float(mlcanopy_inst.zdisp_canopy[p])
    obu_p   = float(mlcanopy_inst.obu_canopy[p])
    beta_p  = float(mlcanopy_inst.beta_canopy[p])
    PrSc_p  = float(mlcanopy_inst.PrSc_canopy[p])
    zref_p  = float(mlcanopy_inst.zref_forcing[p])

    # psi and psihat at canopy top — Fortran lines 475-476
    _, _, psim_hc, psim_hat_hc = _GetPsiRSL(
        zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p
    )

    hc_minus_d = ztop_p - zdisp_p
    exp1 = math.exp(-vkc / beta_p)
    exp2 = math.exp(psim_hat_hc)

    def _z0m_func(trial: float) -> float:
        """Implicit equation: z0m_formula(trial) - trial."""
        psim_t = _psim_monin_obukhov(trial / obu_p)
        return hc_minus_d * exp1 * math.exp(-psim_hc + psim_t) * exp2 - trial

    aval: float = ztop_p
    bval: float = 0.0
    err:  float = 0.001
    nmax: int   = 20

    fa = _z0m_func(aval)
    fb = _z0m_func(bval)

    if fa * fb > 0.0:
        endrun(msg=' ERROR: RoughnessLength: bisection error - f(a) and f(b) do not have opposite signs')

    cval: float = 0.0
    n: int = 1
    while abs(bval - aval) > err and n <= nmax:
        cval = (aval + bval) / 2.0
        fc   = _z0m_func(cval)
        if fa * fc < 0.0:
            bval = cval; fb = fc
        else:
            aval = cval; fa = fc
        n += 1

    if n > nmax:
        endrun(msg=' ERROR: RoughnessLength: maximum iteration exceeded')

    return mlcanopy_inst._replace(
        z0m_canopy = mlcanopy_inst.z0m_canopy.at[p].set(cval)
    )


# ===========================================================================
# Private: WindProfile
# ===========================================================================

def _WindProfile(
    p: int,
    lm_over_beta: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Wind speed profile above and within the canopy.

    Mirrors Fortran subroutine ``WindProfile`` (lines 522-565).

    Reference: Bonan et al. (2018), eqs. (19) and (21).

    **Above canopy** (ic > ntop, at height zs):

    .. code-block:: none

        wind(ic) = ustar/vkc * (log((zs-d)/(ztop-d)) + psim)

    **At canopy top**:

    .. code-block:: none

        uaf = ustar / beta

    **Within canopy** (ic ≤ ntop, at height zs):

    .. code-block:: none

        wind(ic) = uaf * exp((zs - ztop) / (lm/beta))

    Args:
        p: Patch index.
        lm_over_beta: lm/β turbulence length scale (m).
        mlcanopy_inst: Canopy container; ``uaf_canopy`` and
            ``wind_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    _ntop  = int(mlcanopy_inst.ntop_canopy[p])
    _ncan  = int(mlcanopy_inst.ncan_canopy[p])
    ztop_p = float(mlcanopy_inst.ztop_canopy[p])
    zdisp_p = float(mlcanopy_inst.zdisp_canopy[p])
    obu_p   = float(mlcanopy_inst.obu_canopy[p])
    beta_p  = float(mlcanopy_inst.beta_canopy[p])
    PrSc_p  = float(mlcanopy_inst.PrSc_canopy[p])
    ustar_p = float(mlcanopy_inst.ustar_canopy[p])

    wind = mlcanopy_inst.wind_profile
    uaf  = mlcanopy_inst.uaf_canopy

    # Pre-extract zs_profile row to avoid per-layer JAX syncs
    _zs_p = np.asarray(mlcanopy_inst.zs_profile[p])

    # Accumulate wind values in numpy array; batch write-back at end
    _wind_new = np.zeros(_ncan + 2)

    # Above-canopy wind — Fortran lines 544-547
    for ic in range(_ntop + 1, _ncan + 1):
        zs_ic = float(_zs_p[ic])
        psim, _, _, _ = _GetPsiRSL(zs_ic, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        zlog_m = math.log((zs_ic - zdisp_p) / (ztop_p - zdisp_p))
        _wind_new[ic] = ustar_p / vkc * (zlog_m + psim)

    # Wind at canopy top — Fortran line 549
    uaf_val = ustar_p / beta_p
    uaf = uaf.at[p].set(uaf_val)

    # Within-canopy wind — Fortran lines 551-553
    for ic in range(1, _ntop + 1):
        zs_ic = float(_zs_p[ic])
        _wind_new[ic] = uaf_val * math.exp((zs_ic - ztop_p) / lm_over_beta)

    # Batch write-back
    _sl = slice(1, _ncan + 1)
    wind = wind.at[p, _sl].set(jnp.array(_wind_new[_sl]))

    return mlcanopy_inst._replace(
        uaf_canopy   = uaf,
        wind_profile = wind,
    )


# ===========================================================================
# Private: AerodynamicConductance
# ===========================================================================

def _AerodynamicConductance(
    p: int,
    lm_over_beta: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Aerodynamic conductances above and within the canopy.

    Mirrors Fortran subroutine ``AerodynamicConductance`` (lines 567-680).

    Reference: Bonan et al. (2018), eqs. (24), (26), (27).

    **Above canopy** (Fortran lines 591-620, eq. 24):

    .. code-block:: none

        gac(ic) = rhomol * vkc * ustar / (log((zs_{i+1}-d)/(zs_i-d)) + psic)

    Special cases: top layer uses ``zref`` as upper boundary; top foliage
    layer uses ``ztop`` as lower boundary and ``zs(ntop+1)`` as upper.
    A consistency check against ``gac_to_hc`` is applied.

    **Within canopy** (Fortran lines 622-641, eq. 26):

    .. code-block:: none

        res  = PrSc / (beta*ustar) * (exp(-zl/lm_beta) - exp(-zu/lm_beta))
        gac  = rhomol / res

    Top foliage layer combines within and above contributions in series.

    **Soil surface conductance** (Fortran lines 643-672):
    Selected by ``HF_extension_type``:

    - **1**: exponential profile extended to ``z0cg = 0.1*z0mg``.
    - **2**: logarithmic profile to ``z0mg`` with wind floor of 0.1 m/s.

    All conductances are capped so that the implied resistance does not
    exceed ``ra_max`` s/m.

    **Eddy diffusivity** (Fortran lines 674-678):

    .. code-block:: none

        kc_eddy(ic) = gac(ic) / rhomol * (zs_{ic+1} - zs_ic)

    Args:
        p: Patch index.
        lm_over_beta: lm/β turbulence length scale (m).
        mlcanopy_inst: Canopy container; ``gac0_soil``, ``gac_profile``,
            and ``kc_eddy_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    _ntop   = int(mlcanopy_inst.ntop_canopy[p])
    _ncan   = int(mlcanopy_inst.ncan_canopy[p])
    ztop_p  = float(mlcanopy_inst.ztop_canopy[p])
    zdisp_p = float(mlcanopy_inst.zdisp_canopy[p])
    obu_p   = float(mlcanopy_inst.obu_canopy[p])
    beta_p  = float(mlcanopy_inst.beta_canopy[p])
    PrSc_p  = float(mlcanopy_inst.PrSc_canopy[p])
    ustar_p = float(mlcanopy_inst.ustar_canopy[p])
    rhomol_p = float(mlcanopy_inst.rhomol_forcing[p])
    zref_p   = float(mlcanopy_inst.zref_forcing[p])
    gac_to_hc_p = float(mlcanopy_inst.gac_to_hc_canopy[p])

    gac   = mlcanopy_inst.gac_profile
    gac0  = mlcanopy_inst.gac0_soil
    kc_eddy = mlcanopy_inst.kc_eddy_profile

    # Pre-extract zs_profile row to avoid per-layer JAX syncs
    _zs_p = np.asarray(mlcanopy_inst.zs_profile[p])

    # Numpy accumulators for batch write-back
    _gac_new    = np.zeros(_ncan + 2)
    _kc_eddy_new = np.zeros(_ncan + 2)

    # ------------------------------------------------------------------
    # Above-canopy conductances — Fortran lines 591-604
    # ------------------------------------------------------------------
    for ic in range(_ntop + 1, _ncan):             # Fortran: ntop+1 to ncan-1
        zs_lo = float(_zs_p[ic])
        zs_hi = float(_zs_p[ic + 1])
        _, psic1, _, _ = _GetPsiRSL(zs_lo, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        _, psic2, _, _ = _GetPsiRSL(zs_hi, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        psic   = psic2 - psic1
        zlog_c = math.log((zs_hi - zdisp_p) / (zs_lo - zdisp_p))
        _gac_new[ic] = rhomol_p * vkc * ustar_p / (zlog_c + psic)

    # Top layer to reference height — Fortran lines 606-612
    ic = _ncan
    zs_lo = float(_zs_p[ic])
    _, psic1, _, _ = _GetPsiRSL(zs_lo, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _, psic2, _, _ = _GetPsiRSL(zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic   = psic2 - psic1
    zlog_c = math.log((zref_p - zdisp_p) / (zs_lo - zdisp_p))
    _gac_new[ic] = rhomol_p * vkc * ustar_p / (zlog_c + psic)

    # ztop to zs(ntop+1) — Fortran lines 614-620
    ic = _ntop
    _, psic1, _, _ = _GetPsiRSL(ztop_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _, psic2, _, _ = _GetPsiRSL(float(_zs_p[ic + 1]), ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic   = psic2 - psic1
    zlog_c = math.log((float(_zs_p[ic + 1]) - zdisp_p) / (ztop_p - zdisp_p))
    gac_above_foliage = rhomol_p * vkc * ustar_p / (zlog_c + psic)

    # Consistency check — Fortran lines 622-626
    sumres = 1.0 / gac_above_foliage
    for ic in range(_ntop + 1, _ncan + 1):
        sumres += 1.0 / _gac_new[ic]
    if abs(1.0 / sumres - gac_to_hc_p) > 1.0e-6:
        endrun(msg=' ERROR: AerodynamicConductance: above-canopy aerodynamic conductance error')

    # ------------------------------------------------------------------
    # Within-canopy conductances — Fortran lines 632-647
    # ------------------------------------------------------------------
    for ic in range(1, _ntop):                     # Fortran: 1 to ntop-1
        zl = float(_zs_p[ic])     - ztop_p
        zu = float(_zs_p[ic + 1]) - ztop_p
        res = (PrSc_p / (beta_p * ustar_p)
               * (math.exp(-zl / lm_over_beta) - math.exp(-zu / lm_over_beta)))
        _gac_new[ic] = rhomol_p / res

    # Top foliage layer: combine below and above — Fortran lines 649-656
    ic = _ntop
    zl  = float(_zs_p[ic]) - ztop_p
    zu  = ztop_p - ztop_p                          # = 0
    res = (PrSc_p / (beta_p * ustar_p)
           * (math.exp(-zl / lm_over_beta) - math.exp(-zu / lm_over_beta)))
    gac_below_foliage = rhomol_p / res
    _gac_new[ic] = 1.0 / (1.0 / gac_below_foliage + 1.0 / gac_above_foliage)

    # ------------------------------------------------------------------
    # Soil surface aerodynamic conductance — Fortran lines 658-686
    # ------------------------------------------------------------------
    z0cg  = 0.1 * z0mg
    zs1   = float(_zs_p[1])
    if z0mg > zs1 or z0cg > zs1:
        endrun(msg=' ERROR: AerodynamicConductance: soil roughness error')

    if HF_extension_type == 1:                     # Fortran lines 668-673
        zl  = z0cg - ztop_p
        zu  = zs1  - ztop_p
        res = (PrSc_p / (beta_p * ustar_p)
               * (math.exp(-zl / lm_over_beta) - math.exp(-zu / lm_over_beta)))
        _gac0_val = rhomol_p / res

    elif HF_extension_type == 2:                   # Fortran lines 675-678
        zlog_m  = math.log(zs1 / z0mg)
        # wind_profile[p,1] was just set by _WindProfile; re-read from mlcanopy_inst
        ustar_g = max(float(mlcanopy_inst.wind_profile[p, 1]), 0.1) * vkc / zlog_m
        _gac0_val = rhomol_p * vkc * ustar_g / zlog_m
    else:
        _gac0_val = float(gac0[p])

    # Resistance cap (ra_max) — Fortran lines 688-694
    _gac0_val = rhomol_p / min(rhomol_p / _gac0_val, ra_max)

    for ic in range(1, _ncan + 1):
        _gac_new[ic] = rhomol_p / min(rhomol_p / _gac_new[ic], ra_max)

    # Eddy diffusivity — Fortran lines 696-700
    for ic in range(1, _ncan + 1):
        if ic == _ncan:
            dz = zref_p - float(_zs_p[ic])
        else:
            dz = float(_zs_p[ic + 1]) - float(_zs_p[ic])
        _kc_eddy_new[ic] = _gac_new[ic] / rhomol_p * dz

    # Batch write-back — 3 bulk ops instead of ~20 per-element writes
    _sl = slice(1, _ncan + 1)
    return mlcanopy_inst._replace(
        gac0_soil       = gac0.at[p].set(_gac0_val),
        gac_profile     = gac.at[p, _sl].set(jnp.array(_gac_new[_sl])),
        kc_eddy_profile = kc_eddy.at[p, _sl].set(jnp.array(_kc_eddy_new[_sl])),
    )


# ===========================================================================
# Private: HF2008 main solver
# ===========================================================================

def _HF2008(
    nstep_ml: int,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Canopy turbulence, wind profile, and aerodynamic conductances using
    the Harman and Finnigan (2008) roughness sublayer parameterisation.

    Mirrors Fortran subroutine ``HF2008`` (lines 67-160).

    References: Bonan et al. (2018) *Geosci. Model Dev.*, 11, 1467-1496;
    Bonan et al. (2025).

    For each patch (Fortran lines 108-156):

    1. Canopy density length scale ``Lc = ztop / (cd*(lai+sai))``.
    2. Temperature and specific humidity at canopy top for
       :func:`_ObuFunc`.
    3. Obukhov length via :func:`_GetObu`.
    4. Roughness length via :func:`_RoughnessLength`.
    5. Compute ``lm = 2*β³*Lc``, ``η = β/lm*ztop``, warn if ``η ≥ η_max``,
       then ``lm/β = ztop/η``.
    6. Wind profile via :func:`_WindProfile`.
    7. Momentum flux profile (Fortran lines 138-144):
       ``mflx(ic) = -ustar²`` above canopy; exponential decay within.
    8. Aerodynamic conductances via :func:`_AerodynamicConductance`.

    Args:
        nstep_ml: Current multilayer timestep counter (for warning messages).
        num_filter: Number of patches.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all turbulence output fields
            updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    from clm_src_utils.clm_time_manager import get_nstep         # noqa: F401
    nstep = get_nstep()

    Lc   = mlcanopy_inst.Lc_canopy
    taf  = mlcanopy_inst.taf_canopy
    qaf  = mlcanopy_inst.qaf_canopy
    mflx = mlcanopy_inst.mflx_profile

    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1])

        pref_p  = float(mlcanopy_inst.pref_forcing[p])
        ztop_p  = float(mlcanopy_inst.ztop_canopy[p])
        lai_p   = float(mlcanopy_inst.lai_canopy[p])
        sai_p   = float(mlcanopy_inst.sai_canopy[p])
        _ntop   = int(mlcanopy_inst.ntop_canopy[p])
        _ncan   = int(mlcanopy_inst.ncan_canopy[p])

        # Canopy density length scale — Fortran line 110
        Lc_val = ztop_p / (cd * (lai_p + sai_p))
        Lc = Lc.at[p].set(Lc_val)
        mlcanopy_inst = mlcanopy_inst._replace(Lc_canopy = Lc)

        # Temperature and humidity at canopy top — Fortran lines 112-113
        tair_ntop = float(mlcanopy_inst.tair_profile[p, _ntop])
        eair_ntop = float(mlcanopy_inst.eair_profile[p, _ntop])
        taf_val = tair_ntop
        qaf_val = (mmh2o / mmdry * eair_ntop
                   / (pref_p - (1.0 - mmh2o / mmdry) * eair_ntop))
        taf = taf.at[p].set(taf_val)
        qaf = qaf.at[p].set(qaf_val)
        mlcanopy_inst = mlcanopy_inst._replace(taf_canopy = taf, qaf_canopy = qaf)

        # Obukhov length — Fortran line 115
        mlcanopy_inst = _GetObu(p, mlcanopy_inst)

        # Roughness length — Fortran line 118
        mlcanopy_inst = _RoughnessLength(p, mlcanopy_inst)

        # lm / beta — Fortran lines 126-130
        beta_p = float(mlcanopy_inst.beta_canopy[p])
        lm     = 2.0 * beta_p ** 3 * Lc_val
        eta    = beta_p / lm * ztop_p
        if eta >= eta_max:
            print(f' Warning: HF2008: lm/beta error')
            print(f' nstep = {nstep}  nstep_ml = {nstep_ml}')
            print(f' eta = {eta}')
        lm_over_beta = ztop_p / eta

        # Wind profile — Fortran line 132
        mlcanopy_inst = _WindProfile(p, lm_over_beta, mlcanopy_inst)

        # Momentum flux profile — Fortran lines 134-140
        ustar_p = float(mlcanopy_inst.ustar_canopy[p])
        _zw_p = np.asarray(mlcanopy_inst.zw_profile[p])
        _mflx_new = np.zeros(_ncan + 2)
        for ic in range(1, _ncan + 1):
            zw_ic = float(_zw_p[ic])
            if zw_ic > ztop_p:
                _mflx_new[ic] = -(ustar_p ** 2)
            else:
                _mflx_new[ic] = (-(ustar_p ** 2)
                                  * math.exp(2.0 * (zw_ic - ztop_p) / lm_over_beta))
        _sl_m = slice(1, _ncan + 1)
        mflx = mflx.at[p, _sl_m].set(jnp.array(_mflx_new[_sl_m]))
        mlcanopy_inst = mlcanopy_inst._replace(mflx_profile = mflx)

        # Aerodynamic conductances — Fortran line 142
        mlcanopy_inst = _AerodynamicConductance(p, lm_over_beta, mlcanopy_inst)

    return mlcanopy_inst


# ===========================================================================
# Public: CanopyTurbulence driver
# ===========================================================================

def CanopyTurbulence(
    nstep_ml: int,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Main driver for the canopy turbulence parameterisation.

    Mirrors Fortran subroutine ``CanopyTurbulence`` (lines 43-67).

    Currently only ``turb_type == 1`` (Harman & Finnigan 2008 RSL) is
    supported; any other value is fatal.

    Args:
        nstep_ml: Current multilayer timestep counter.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all turbulence output fields
            updated by :func:`_HF2008`.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if turb_type == 1:
        return _HF2008(nstep_ml, num_filter, filter_patch, mlcanopy_inst)
    else:
        endrun(msg=' ERROR: CanopyTurbulence: turb_type not valid')
        return mlcanopy_inst    # Unreachable


# ===========================================================================
# Public: LookupPsihatINI
# ===========================================================================

def LookupPsihatINI() -> None:
    """
    Initialise the RSL psihat look-up tables from a NetCDF file.

    Mirrors Fortran subroutine ``LookupPsihatINI`` (lines 455-530).

    Reads variables ``dtLgridM``, ``zdtgridM``, ``psigridM``,
    ``dtLgridH``, ``zdtgridH``, and ``psigridH`` from the file whose
    path is given by the module-level variable ``rslfile``.

    Fortran stores the grids as:

    .. code-block:: none

        zdtgridM(nZ, 1), dtLgridM(1, nL), psigridM(nZ, nL)

    while the NetCDF file has dimensions in the opposite order:

    .. code-block:: none

        psigridM_nc(nL, nZ)  →  psigridM[ii, jj] = psigridM_nc[jj, ii]

    The copy loop transposes the NetCDF arrays into the Fortran (and
    Python) convention.  After this routine the module-level arrays
    ``zdtgridM``, ``dtLgridM``, ``psigridM``, ``zdtgridH``,
    ``dtLgridH``, and ``psigridH`` are populated and ready for use
    by :func:`_LookupPsihat`.

    Raises:
        SystemExit: (via :func:`endrun`) if the NetCDF dimensions do
            not match the expected ``nZ`` and ``nL`` or if any variable
            cannot be read.
    """
    global dtLgridM, dtLgridH, zdtgridM, zdtgridH, psigridM, psigridH
    global _zdtgM_1d, _dtLgM_1d, _neg_zdtgM_1d
    global _zdtgH_1d, _dtLgH_1d, _neg_zdtgH_1d
    
    import netCDF4 as nc                           # noqa: F401 — defer import

    print(f'Attempting to read RSL look-up table .....')

    with nc.Dataset(rslfile, 'r') as ncid:

        # Check dimensions — Fortran lines 487-498
        nZ_nc = len(ncid.dimensions['nZ'])
        nL_nc = len(ncid.dimensions['nL'])
        if nZ_nc != nZ:
            endrun(msg=' ERROR: LookupPsihatINI: nZ does not equal expected value')
        if nL_nc != nL:
            endrun(msg=' ERROR: LookupPsihatINI: nL does not equal expected value')

        # Read 1-D coordinate arrays and 2-D psihat grids — Fortran lines 500-520
        def _read(name: str):
            if name not in ncid.variables:
                endrun(msg=f' ERROR: LookupPsihatINI: error reading {name}')
            return ncid.variables[name][:]

        dtLgridM_nc = np.asarray(_read('dtLgridM'), dtype=float)   # shape (nL,)
        zdtgridM_nc = np.asarray(_read('zdtgridM'), dtype=float)   # shape (nZ,)
        psigridM_nc = np.asarray(_read('psigridM'), dtype=float)   # Actually shape (nZ, nL) not (nL, nZ)!

        dtLgridH_nc = np.asarray(_read('dtLgridH'), dtype=float)
        zdtgridH_nc = np.asarray(_read('zdtgridH'), dtype=float)
        psigridH_nc = np.asarray(_read('psigridH'), dtype=float)

    print(f'Successfully read RSL look-up table')

    # Copy into module-level arrays with Fortran index conventions —
    # Fortran lines 522-530:
    #   dtLgridM(1, jj) = dtLgridM_nc(jj)  →  dtLgridM[0, jj]
    #   zdtgridM(ii, 1) = zdtgridM_nc(ii)  →  zdtgridM[ii, 0]
    #   psigridM(ii,jj) = psigridM_nc(jj,ii) →  psigridM[ii, jj]
    # 
    # NOTE: psigridM_nc is already in shape (nZ, nL), so no transpose needed
    
    # Populate numpy arrays directly (no JAX immutable operations needed)
    dtLgridM[0, :] = dtLgridM_nc
    dtLgridH[0, :] = dtLgridH_nc
    zdtgridM[:, 0] = zdtgridM_nc
    zdtgridH[:, 0] = zdtgridH_nc
    psigridM[:, :] = psigridM_nc
    psigridH[:, :] = psigridH_nc

    # Populate cached 1D views — eliminates per-call slicing and negation in _LookupPsihatM/H
    _dtLgM_1d     = dtLgridM[0].copy()
    _zdtgM_1d     = zdtgridM[:, 0].copy()
    _neg_zdtgM_1d = -_zdtgM_1d
    _dtLgH_1d     = dtLgridH[0].copy()
    _zdtgH_1d     = zdtgridH[:, 0].copy()
    _neg_zdtgH_1d = -_zdtgH_1d