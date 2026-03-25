"""
JAX translation of MLLeafPhotosynthesisMod Fortran module.

Leaf photosynthesis and stomatal conductance for the multilayer
canopy model.  Provides one public routine (:func:`LeafPhotosynthesis`)
and the following private helpers:

- :func:`_ft`: Arrhenius activation temperature response.
- :func:`_fth`: High-temperature deactivation factor.
- :func:`_fth25`: Deactivation scaling factor at 25 °C.
- :func:`_CiFunc`: Photosynthesis + gs for a specified Ci (hybrid callback).
- :func:`_CiFuncGs`: Photosynthesis for a specified gs.
- :func:`_StomataOptimization`: WUE-optimal stomatal conductance.
- :func:`_StomataEfficiency`: Marginal WUE check (zbrent/bisection callback).
- :func:`_RealizedRate`: Minimum or co-limited gross photosynthesis.

Original Fortran module: MLLeafPhotosynthesisMod
Fortran lines 1-500
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from clm_src_main.abortutils import endrun                       # noqa: F401
from clm_src_main.clm_varctl import iulog                       # noqa: F401
from clm_src_main.clm_varcon import tfrz                        # noqa: F401
from clm_src_main.PatchType import patch                        # noqa: F401
from clm_src_main.pftconMod import pftcon                       # noqa: F401
from multilayer_canopy.MLclm_varcon import (                         # noqa: F401
    rgas,
    kc25, ko25, cp25, kcha, koha, cpha,
    vcmaxha_noacclim, vcmaxha_acclim,
    jmaxha_noacclim,  jmaxha_acclim,
    vcmaxhd_noacclim, vcmaxhd_acclim,
    jmaxhd_noacclim,  jmaxhd_acclim,
    vcmaxse_noacclim, vcmaxse_acclim,
    jmaxse_noacclim,  jmaxse_acclim,
    rdha, rdhd, rdse,
    phi_psII, theta_j, vpd_min_MED, rh_min_BB,
    dh2o_to_dco2, qe_c4,
    colim_c3a, colim_c4a, colim_c4b,
)
from multilayer_canopy.MLclm_varctl import (                         # noqa: F401
    gs_type, acclim_type, gspot_type, colim_type, gs_solver,
)
from multilayer_canopy.MLpftconMod import MLpftcon                   # noqa: F401
from multilayer_canopy.MLMathToolsMod import (hybrid, quadratic, zbrent, bisection,  # noqa: F401
                                              hybrid_scalar, zbrent_scalar, bisection_scalar)
from multilayer_canopy.MLWaterVaporMod import SatVap                 # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type       # noqa: F401


# ---------------------------------------------------------------------------
# Private: temperature response functions
# ---------------------------------------------------------------------------

def _ft(tl: float, ha: float) -> float:
    """
    Arrhenius activation temperature response for photosynthesis.

    Mirrors Fortran function ``ft`` (private, lines 35-48).

    .. code-block:: none

        ans = exp( ha / (rgas*(tfrz+25)) * (1 - (tfrz+25)/tl) )

    Args:
        tl: Leaf temperature (K).
        ha: Activation energy (J/mol).

    Returns:
        Dimensionless temperature response factor.
    """
    return math.exp(ha / (rgas * (tfrz + 25.0)) * (1.0 - (tfrz + 25.0) / tl))


def _fth(tl: float, hd: float, se: float, c: float) -> float:
    """
    High-temperature deactivation factor for photosynthesis.

    Mirrors Fortran function ``fth`` (private, lines 50-67).

    .. code-block:: none

        ans = c / (1 + exp((-hd + se*tl) / (rgas*tl)))

    Args:
        tl: Leaf temperature (K).
        hd: Deactivation energy (J/mol).
        se: Entropy term (J/mol/K).
        c:  Scaling factor (= 1.0 at 25 °C).

    Returns:
        Dimensionless deactivation factor.
    """
    return c / (1.0 + math.exp((-hd + se * tl) / (rgas * tl)))


def _fth25(hd: float, se: float) -> float:
    """
    Scaling factor for deactivation such that :func:`_fth` = 1 at 25 °C.

    Mirrors Fortran function ``fth25`` (private, lines 69-85).

    .. code-block:: none

        ans = 1 + exp((-hd + se*(tfrz+25)) / (rgas*(tfrz+25)))

    Args:
        hd: Deactivation energy (J/mol).
        se: Entropy term (J/mol/K).

    Returns:
        Dimensionless scaling factor ``c`` for :func:`_fth`.
    """
    t25 = tfrz + 25.0
    return 1.0 + math.exp((-hd + se * t25) / (rgas * t25))


# ---------------------------------------------------------------------------
# Private: minimum or co-limited photosynthesis
# ---------------------------------------------------------------------------

def _RealizedRate(
    c3psn_val: float,
    ac: float,
    aj: float,
    ap: float,
) -> float:
    """
    Return the realized (gross) photosynthesis rate as the minimum or
    co-limited combination of the three limiting rates.

    Mirrors Fortran subroutine ``RealizedRate`` (private, lines 450-500).

    ``colim_type == 0``: simple minimum of rates.
    ``colim_type == 1``: quadratic co-limitation of Ac and Aj first,
    then a second co-limitation with Ap for C4.

    Args:
        c3psn_val: Photosynthetic pathway flag (1 = C3, 0 = C4).
        ac: Rubisco-limited gross photosynthesis (umol CO2/m2/s).
        aj: RuBP-regeneration-limited gross photosynthesis (umol CO2/m2/s).
        ap: Product-limited (C3) or CO2-limited (C4) gross photosynthesis.

    Returns:
        ``agross``: Gross photosynthesis (umol CO2/m2/s).
    """
    is_c3 = round(c3psn_val) == 1                  # Fortran: nint(c3psn) == 1

    if colim_type == 0:                            # Fortran lines 465-474
        if is_c3:
            return min(ac, aj)
        else:
            return min(ac, aj, ap)

    elif colim_type == 1:                          # Fortran lines 476-496
        aquad = colim_c3a if is_c3 else colim_c4a
        bquad = -(ac + aj)
        cquad = ac * aj
        r1, r2 = quadratic(aquad, bquad, cquad)
        ai = min(r1, r2)
        if is_c3:
            return ai
        else:
            aquad = colim_c4b
            bquad = -(ai + ap)
            cquad = ai * ap
            r1, r2 = quadratic(aquad, bquad, cquad)
            return min(r1, r2)

    else:
        endrun(msg=' ERROR: RealizedRate: colim_type not valid')
        return 0.0    # Unreachable


# ---------------------------------------------------------------------------
# Private: photosynthesis for a specified stomatal conductance
# ---------------------------------------------------------------------------

def _CiFuncGs(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> Tuple[float, mlcanopy_type]:
    """
    Calculate leaf photosynthesis for the current stomatal conductance
    ``gs(p,ic,il)`` and return the intercellular CO2 ``ci_val``
    derived from the diffusion equation.

    Mirrors Fortran subroutine ``CiFuncGs`` (private, lines 320-400).

    Substitutes the diffusion equation for Ci into each metabolic
    photosynthesis expression, yielding a quadratic in An:

    .. code-block:: none

        a*An^2 + b*An + c = 0     →    An = min(r1, r2) + rd

    For C4, the PEP carboxylase-limited rate is computed directly.
    After finding the limiting rate via :func:`_RealizedRate`,
    ``ci_val`` is recovered from the diffusion equation.

    Args:
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit/shaded leaf index.
        mlcanopy_inst: Canopy container; ac, aj, ap, agross, anet,
            cs updated.

    Returns:
        Tuple ``(ci_val, mlcanopy_inst)``.
    """
    c3psn  = pftcon.c3psn
    pft    = int(patch.itype[p])
    is_c3  = round(float(c3psn[pft])) == 1

    dpai_ic = float(mlcanopy_inst.dpai_profile[p, ic])
    ac  = mlcanopy_inst.ac_leaf
    aj  = mlcanopy_inst.aj_leaf
    ap  = mlcanopy_inst.ap_leaf
    agross = mlcanopy_inst.agross_leaf
    anet   = mlcanopy_inst.anet_leaf
    cs     = mlcanopy_inst.cs_leaf

    if dpai_ic > 0.0:                              # Fortran lines 378-418
        gbc_ic = float(mlcanopy_inst.gbc_leaf[p, ic, il])
        gs_ic  = float(mlcanopy_inst.gs_leaf[p, ic, il])
        cair_ic = float(mlcanopy_inst.cair_profile[p, ic])
        vcmax_ic = float(mlcanopy_inst.vcmax_leaf[p, ic, il])
        je_ic    = float(mlcanopy_inst.je_leaf[p, ic, il])
        kp_ic    = float(mlcanopy_inst.kp_leaf[p, ic, il])
        rd_ic    = float(mlcanopy_inst.rd_leaf[p, ic, il])
        kc_ic    = float(mlcanopy_inst.kc_leaf[p, ic, il])
        ko_ic    = float(mlcanopy_inst.ko_leaf[p, ic, il])
        cp_ic    = float(mlcanopy_inst.cp_leaf[p, ic, il])
        o2ref_p  = float(mlcanopy_inst.o2ref_forcing[p])
        apar_ic  = float(mlcanopy_inst.apar_leaf[p, ic, il])

        gleaf = 1.0 / (1.0 / gbc_ic + dh2o_to_dco2 / gs_ic)

        if is_c3:
            # C3 Rubisco-limited — Fortran lines 386-392
            a0 = vcmax_ic
            b0 = kc_ic * (1.0 + o2ref_p / ko_ic)
            aq = 1.0 / gleaf
            bq = -(cair_ic + b0) - (a0 - rd_ic) / gleaf
            cq = a0 * (cair_ic - cp_ic) - rd_ic * (cair_ic + b0)
            r1, r2 = quadratic(aq, bq, cq)
            ac_val = min(r1, r2) + rd_ic

            # C3 RuBP-regeneration-limited — Fortran lines 394-400
            a0 = je_ic / 4.0
            b0 = 2.0 * cp_ic
            aq = 1.0 / gleaf
            bq = -(cair_ic + b0) - (a0 - rd_ic) / gleaf
            cq = a0 * (cair_ic - cp_ic) - rd_ic * (cair_ic + b0)
            r1, r2 = quadratic(aq, bq, cq)
            aj_val = min(r1, r2) + rd_ic

            ap_val = 0.0                           # Fortran line 402

        else:
            # C4 — Fortran lines 404-409
            ac_val = vcmax_ic
            aj_val = qe_c4 * apar_ic
            ap_val = kp_ic * (cair_ic * gleaf + rd_ic) / (gleaf + kp_ic)

        agross_val = _RealizedRate(float(c3psn[pft]), ac_val, aj_val, ap_val)
        anet_val   = agross_val - rd_ic

        cs_val = max(cair_ic - anet_val / gbc_ic, 1.0)
        ci_val = cair_ic - anet_val / gleaf

        ac     = ac.at[p, ic, il].set(ac_val)
        aj     = aj.at[p, ic, il].set(aj_val)
        ap     = ap.at[p, ic, il].set(ap_val)
        agross = agross.at[p, ic, il].set(agross_val)
        anet   = anet.at[p, ic, il].set(anet_val)
        cs     = cs.at[p, ic, il].set(cs_val)

    else:                                          # Fortran lines 420-427
        ac     = ac.at[p, ic, il].set(0.0)
        aj     = aj.at[p, ic, il].set(0.0)
        ap     = ap.at[p, ic, il].set(0.0)
        agross = agross.at[p, ic, il].set(0.0)
        anet   = anet.at[p, ic, il].set(0.0)
        cs     = cs.at[p, ic, il].set(0.0)
        ci_val = 0.0

    mlcanopy_inst = mlcanopy_inst._replace(
        ac_leaf     = ac,
        aj_leaf     = aj,
        ap_leaf     = ap,
        agross_leaf = agross,
        anet_leaf   = anet,
        cs_leaf     = cs,
    )
    return ci_val, mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: pure-scalar version of _CiFuncGs (no JAX reads/writes)
# ---------------------------------------------------------------------------

def _CiFuncGsPure(
    gs_val: float,
    *,
    is_c3: bool,
    dpai_ic: float,
    gbc_ic: float,
    cair_ic: float,
    vcmax_ic: float,
    je_ic: float,
    kp_ic: float,
    rd_ic: float,
    kc_ic: float,
    ko_ic: float,
    cp_ic: float,
    o2ref_p: float,
    apar_ic: float,
    c3psn_pft_val: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Pure-scalar version of :func:`_CiFuncGs`.

    All inputs are pre-extracted Python floats; no JAX array reads or
    writes occur.  Returns ``(ci_val, ac_val, aj_val, ap_val,
    agross_val, anet_val, cs_val)`` as Python floats.
    """
    if dpai_ic > 0.0:
        gleaf = 1.0 / (1.0 / gbc_ic + dh2o_to_dco2 / gs_val)

        if is_c3:
            a0 = vcmax_ic
            b0 = kc_ic * (1.0 + o2ref_p / ko_ic)
            aq = 1.0 / gleaf
            bq = -(cair_ic + b0) - (a0 - rd_ic) / gleaf
            cq = a0 * (cair_ic - cp_ic) - rd_ic * (cair_ic + b0)
            r1, r2 = quadratic(aq, bq, cq)
            ac_val = min(r1, r2) + rd_ic

            a0 = je_ic / 4.0
            b0 = 2.0 * cp_ic
            aq = 1.0 / gleaf
            bq = -(cair_ic + b0) - (a0 - rd_ic) / gleaf
            cq = a0 * (cair_ic - cp_ic) - rd_ic * (cair_ic + b0)
            r1, r2 = quadratic(aq, bq, cq)
            aj_val = min(r1, r2) + rd_ic

            ap_val = 0.0
        else:
            ac_val = vcmax_ic
            aj_val = qe_c4 * apar_ic
            ap_val = kp_ic * (cair_ic * gleaf + rd_ic) / (gleaf + kp_ic)

        agross_val = _RealizedRate(c3psn_pft_val, ac_val, aj_val, ap_val)
        anet_val   = agross_val - rd_ic
        cs_val     = max(cair_ic - anet_val / gbc_ic, 1.0)
        ci_val     = cair_ic - anet_val / gleaf

    else:
        ac_val = aj_val = ap_val = agross_val = anet_val = cs_val = ci_val = 0.0

    return ci_val, ac_val, aj_val, ap_val, agross_val, anet_val, cs_val


# ---------------------------------------------------------------------------
# Private: photosynthesis + gs for a specified Ci (hybrid callback)
# ---------------------------------------------------------------------------

def _CiFunc(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    ci_val: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Calculate leaf photosynthesis and stomatal conductance for a
    specified ``ci_val``, then derive a new Ci from the diffusion
    equation.  Returns ``ci_dif = cinew - ci_val`` (= 0 at convergence).

    Mirrors Fortran subroutine ``CiFunc`` (private, lines 228-320).

    This is the callback function passed to :func:`hybrid`.

    **Step 1** — metabolic (demand-side) photosynthesis at ``ci_val``.
    **Step 2** — stomatal conductance from Ball-Berry or Medlyn
    quadratic.
    **Step 3** — new Ci from the diffusion equation.

    Returns ``ci_dif = 0`` whenever ``dpai == 0`` or ``anet < 0``.

    Args:
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit/shaded leaf index.
        mlcanopy_inst: Canopy container; ac, aj, ap, agross, anet,
            cs, gs updated.
        ci_val: Trial value for Ci (umol/mol).

    Returns:
        Tuple ``(ci_dif, mlcanopy_inst)``.
    """
    c3psn = pftcon.c3psn
    pft   = int(patch.itype[p])
    is_c3 = round(float(c3psn[pft])) == 1

    dpai_ic = float(mlcanopy_inst.dpai_profile[p, ic])
    ac  = mlcanopy_inst.ac_leaf
    aj  = mlcanopy_inst.aj_leaf
    ap  = mlcanopy_inst.ap_leaf
    agross = mlcanopy_inst.agross_leaf
    anet   = mlcanopy_inst.anet_leaf
    cs     = mlcanopy_inst.cs_leaf
    gs     = mlcanopy_inst.gs_leaf

    if dpai_ic > 0.0:                              # Fortran lines 266-313

        vcmax_ic = float(mlcanopy_inst.vcmax_leaf[p, ic, il])
        je_ic    = float(mlcanopy_inst.je_leaf[p, ic, il])
        kp_ic    = float(mlcanopy_inst.kp_leaf[p, ic, il])
        rd_ic    = float(mlcanopy_inst.rd_leaf[p, ic, il])
        kc_ic    = float(mlcanopy_inst.kc_leaf[p, ic, il])
        ko_ic    = float(mlcanopy_inst.ko_leaf[p, ic, il])
        cp_ic    = float(mlcanopy_inst.cp_leaf[p, ic, il])
        o2ref_p  = float(mlcanopy_inst.o2ref_forcing[p])
        cair_ic  = float(mlcanopy_inst.cair_profile[p, ic])
        apar_ic  = float(mlcanopy_inst.apar_leaf[p, ic, il])
        gbc_ic   = float(mlcanopy_inst.gbc_leaf[p, ic, il])
        gbv_ic   = float(mlcanopy_inst.gbv_leaf[p, ic, il])
        g0_p     = float(mlcanopy_inst.g0_canopy[p])
        g1_p     = float(mlcanopy_inst.g1_canopy[p])
        ceair_ic = float(mlcanopy_inst.ceair_leaf[p, ic, il])
        lesat_ic = float(mlcanopy_inst.leaf_esat_leaf[p, ic, il])

        # --- Step 1: metabolic photosynthesis — Fortran lines 268-292 ---
        if is_c3:
            ac_val = (vcmax_ic * max(ci_val - cp_ic, 0.0)
                      / (ci_val + kc_ic * (1.0 + o2ref_p / ko_ic)))
            aj_val = (je_ic * max(ci_val - cp_ic, 0.0)
                      / (4.0 * ci_val + 8.0 * cp_ic))
            ap_val = 0.0
        else:
            ac_val = vcmax_ic
            aj_val = qe_c4 * apar_ic
            ap_val = kp_ic * max(ci_val, 0.0)

        agross_val = _RealizedRate(float(c3psn[pft]), ac_val, aj_val, ap_val)
        ac_val     = max(ac_val, 0.0)
        aj_val     = max(aj_val, 0.0)
        ap_val     = max(ap_val, 0.0)
        agross_val = max(agross_val, 0.0)
        anet_val   = agross_val - rd_ic

        cs_val = max(cair_ic - anet_val / gbc_ic, 1.0)  # Fortran line 299

        # --- Step 2: stomatal conductance — Fortran lines 301-322 ---
        if gs_type == 1:                           # Ball-Berry
            if anet_val > 0.0:
                term  = anet_val / cs_val
                aq    = 1.0
                bq    = gbv_ic - g0_p - g1_p * term
                cq    = -gbv_ic * (g0_p + g1_p * term * ceair_ic / lesat_ic)
                r1, r2 = quadratic(aq, bq, cq)
                gs_val = max(r1, r2)
            else:
                gs_val = g0_p

        elif gs_type == 0:                         # Medlyn
            if anet_val > 0.0:
                vpd_term = max(lesat_ic - ceair_ic, vpd_min_MED) * 0.001
                term     = dh2o_to_dco2 * anet_val / cs_val
                aq       = 1.0
                bq       = -(2.0 * (g0_p + term)
                             + (g1_p * term) ** 2 / (gbv_ic * vpd_term))
                cq       = (g0_p * g0_p
                            + (2.0 * g0_p
                               + term * (1.0 - g1_p * g1_p / vpd_term)) * term)
                r1, r2   = quadratic(aq, bq, cq)
                gs_val   = max(r1, r2)
            else:
                gs_val = g0_p
        else:
            gs_val = g0_p    # Fallback (should not reach)

        # --- Step 3: new Ci from diffusion — Fortran lines 324-330 ---
        gleaf  = 1.0 / (1.0 / gbc_ic + dh2o_to_dco2 / gs_val)
        cinew  = cair_ic - anet_val / gleaf
        ci_dif = cinew - ci_val
        if anet_val < 0.0:
            ci_dif = 0.0                           # Fortran line 330

        ac     = ac.at[p, ic, il].set(ac_val)
        aj     = aj.at[p, ic, il].set(aj_val)
        ap     = ap.at[p, ic, il].set(ap_val)
        agross = agross.at[p, ic, il].set(agross_val)
        anet   = anet.at[p, ic, il].set(anet_val)
        cs     = cs.at[p, ic, il].set(cs_val)
        gs     = gs.at[p, ic, il].set(gs_val)

    else:                                          # Fortran lines 332-340
        ac     = ac.at[p, ic, il].set(0.0)
        aj     = aj.at[p, ic, il].set(0.0)
        ap     = ap.at[p, ic, il].set(0.0)
        agross = agross.at[p, ic, il].set(0.0)
        anet   = anet.at[p, ic, il].set(0.0)
        cs     = cs.at[p, ic, il].set(0.0)
        gs     = gs.at[p, ic, il].set(0.0)
        ci_dif = 0.0

    mlcanopy_inst = mlcanopy_inst._replace(
        ac_leaf     = ac,
        aj_leaf     = aj,
        ap_leaf     = ap,
        agross_leaf = agross,
        anet_leaf   = anet,
        cs_leaf     = cs,
        gs_leaf     = gs,
    )
    return ci_dif, mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: pure-scalar version of _CiFunc (no JAX reads/writes)
# ---------------------------------------------------------------------------

def _CiFuncPure(
    ci_val: float,
    *,
    is_c3: bool,
    vcmax_ic: float,
    je_ic: float,
    kp_ic: float,
    rd_ic: float,
    kc_ic: float,
    ko_ic: float,
    cp_ic: float,
    o2ref_p: float,
    cair_ic: float,
    apar_ic: float,
    gbc_ic: float,
    gbv_ic: float,
    g0_p: float,
    g1_p: float,
    ceair_ic: float,
    lesat_ic: float,
    c3psn_pft_val: float,
    dpai_ic: float,
) -> float:
    """
    Pure-scalar version of :func:`_CiFunc`.

    All inputs are pre-extracted Python floats; no JAX array reads or
    writes occur.  Returns ``ci_dif = ci_new - ci_val`` (= 0 at
    convergence) as a Python float.
    """
    if dpai_ic <= 0.0:
        return 0.0

    if is_c3:
        ac_val = (vcmax_ic * max(ci_val - cp_ic, 0.0)
                  / (ci_val + kc_ic * (1.0 + o2ref_p / ko_ic)))
        aj_val = (je_ic * max(ci_val - cp_ic, 0.0)
                  / (4.0 * ci_val + 8.0 * cp_ic))
        ap_val = 0.0
    else:
        ac_val = vcmax_ic
        aj_val = qe_c4 * apar_ic
        ap_val = kp_ic * max(ci_val, 0.0)

    agross_val = _RealizedRate(c3psn_pft_val, ac_val, aj_val, ap_val)
    ac_val     = max(ac_val, 0.0)
    aj_val     = max(aj_val, 0.0)
    ap_val     = max(ap_val, 0.0)
    agross_val = max(agross_val, 0.0)
    anet_val   = agross_val - rd_ic

    cs_val = max(cair_ic - anet_val / gbc_ic, 1.0)

    if gs_type == 1:
        if anet_val > 0.0:
            term  = anet_val / cs_val
            aq    = 1.0
            bq    = gbv_ic - g0_p - g1_p * term
            cq    = -gbv_ic * (g0_p + g1_p * term * ceair_ic / lesat_ic)
            r1, r2 = quadratic(aq, bq, cq)
            gs_val = max(r1, r2)
        else:
            gs_val = g0_p
    elif gs_type == 0:
        if anet_val > 0.0:
            vpd_term = max(lesat_ic - ceair_ic, vpd_min_MED) * 0.001
            term     = dh2o_to_dco2 * anet_val / cs_val
            aq       = 1.0
            bq       = -(2.0 * (g0_p + term)
                         + (g1_p * term) ** 2 / (gbv_ic * vpd_term))
            cq       = (g0_p * g0_p
                        + (2.0 * g0_p
                           + term * (1.0 - g1_p * g1_p / vpd_term)) * term)
            r1, r2   = quadratic(aq, bq, cq)
            gs_val   = max(r1, r2)
        else:
            gs_val = g0_p
    else:
        gs_val = g0_p

    gleaf  = 1.0 / (1.0 / gbc_ic + dh2o_to_dco2 / gs_val)
    cinew  = cair_ic - anet_val / gleaf
    ci_dif = cinew - ci_val
    if anet_val < 0.0:
        ci_dif = 0.0

    return ci_dif


# ---------------------------------------------------------------------------
# Private: marginal water-use efficiency check (zbrent/bisection callback)
# ---------------------------------------------------------------------------

def _StomataEfficiency(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    gs_val: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Marginal water-use efficiency check for optimal stomatal conductance.

    Mirrors Fortran subroutine ``StomataEfficiency`` (private,
    lines 430-450).

    Computes photosynthesis at ``gs_val - delta`` and ``gs_val``,
    then checks whether the marginal gain in An exceeds
    ``iota * vpd * delta``:

    .. code-block:: none

        check = (An_high - An_low) - iota * delta * (vpd / pref)

    Positive ``check`` → increase in gs is still profitable.
    Negative ``check`` → optimal gs has been exceeded.

    This function matches the :data:`MLMathToolsMod.FuncType` callback
    signature used by :func:`zbrent` and :func:`bisection`.

    Args:
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit/shaded leaf index.
        mlcanopy_inst: Canopy container threaded through CiFuncGs calls.
        gs_val: Trial stomatal conductance (mol H2O/m2/s).

    Returns:
        Tuple ``(check, mlcanopy_inst)``.
    """
    delta: float = 0.001    # Fortran: delta = 0.001_r8

    pft = int(patch.itype[p])
    iota  = float(MLpftcon.iota_SPA[pft])
    pref_p = float(mlcanopy_inst.pref_forcing[p])
    eair_ic = float(mlcanopy_inst.eair_profile[p, ic])
    gbv_ic  = float(mlcanopy_inst.gbv_leaf[p, ic, il])
    lesat_ic = float(mlcanopy_inst.leaf_esat_leaf[p, ic, il])

    # Photosynthesis at lower gs — Fortran lines 441-443
    mlcanopy_inst = mlcanopy_inst._replace(
        gs_leaf = mlcanopy_inst.gs_leaf.at[p, ic, il].set(gs_val - delta)
    )
    ci_lo, mlcanopy_inst = _CiFuncGs(p, ic, il, mlcanopy_inst)
    mlcanopy_inst = mlcanopy_inst._replace(
        ci_leaf = mlcanopy_inst.ci_leaf.at[p, ic, il].set(ci_lo)
    )
    an_low = float(mlcanopy_inst.anet_leaf[p, ic, il])

    # Photosynthesis at gs_val — Fortran lines 445-447
    mlcanopy_inst = mlcanopy_inst._replace(
        gs_leaf = mlcanopy_inst.gs_leaf.at[p, ic, il].set(gs_val)
    )
    ci_hi, mlcanopy_inst = _CiFuncGs(p, ic, il, mlcanopy_inst)
    mlcanopy_inst = mlcanopy_inst._replace(
        ci_leaf = mlcanopy_inst.ci_leaf.at[p, ic, il].set(ci_hi)
    )
    an_high = float(mlcanopy_inst.anet_leaf[p, ic, il])

    # VPD at leaf surface — Fortran lines 449-452
    gs_cur = float(mlcanopy_inst.gs_leaf[p, ic, il])
    hs     = ((gbv_ic * eair_ic + gs_cur * lesat_ic)
              / ((gbv_ic + gs_cur) * lesat_ic))
    vpd    = max(lesat_ic - hs * lesat_ic, vpd_min_MED)

    # Marginal WUE check — Fortran line 455
    check = (an_high - an_low) - iota * delta * (vpd / pref_p)

    return check, mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: pure-scalar version of _StomataEfficiency (no JAX reads/writes)
# ---------------------------------------------------------------------------

def _StomataEfficiencyPure(
    gs_val: float,
    *,
    iota: float,
    pref_p: float,
    eair_ic: float,
    gbv_ic: float,
    lesat_ic: float,
    # Forwarded to _CiFuncGsPure:
    is_c3: bool,
    dpai_ic: float,
    gbc_ic: float,
    cair_ic: float,
    vcmax_ic: float,
    je_ic: float,
    kp_ic: float,
    rd_ic: float,
    kc_ic: float,
    ko_ic: float,
    cp_ic: float,
    o2ref_p: float,
    apar_ic: float,
    c3psn_pft_val: float,
) -> float:
    """
    Pure-scalar version of :func:`_StomataEfficiency`.

    All inputs are pre-extracted Python floats; no JAX array reads or
    writes occur.  Returns ``check`` (positive → increase gs is
    profitable, negative → optimal gs exceeded) as a Python float.
    """
    delta: float = 0.001

    _kwargs = dict(
        is_c3=is_c3, dpai_ic=dpai_ic, gbc_ic=gbc_ic, cair_ic=cair_ic,
        vcmax_ic=vcmax_ic, je_ic=je_ic, kp_ic=kp_ic, rd_ic=rd_ic,
        kc_ic=kc_ic, ko_ic=ko_ic, cp_ic=cp_ic, o2ref_p=o2ref_p,
        apar_ic=apar_ic, c3psn_pft_val=c3psn_pft_val,
    )

    _, _, _, _, _, an_low, _ = _CiFuncGsPure(gs_val - delta, **_kwargs)
    _, _, _, _, _, an_high, _ = _CiFuncGsPure(gs_val, **_kwargs)

    hs  = ((gbv_ic * eair_ic + gs_val * lesat_ic)
           / ((gbv_ic + gs_val) * lesat_ic))
    vpd = max(lesat_ic - hs * lesat_ic, vpd_min_MED)

    return (an_high - an_low) - iota * delta * (vpd / pref_p)


# ---------------------------------------------------------------------------
# Private: WUE-optimal stomatal conductance
# ---------------------------------------------------------------------------

def _StomataOptimization(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Solve for the WUE-optimal stomatal conductance.

    Mirrors Fortran subroutine ``StomataOptimization`` (private,
    lines 400-430).

    Sets ``gs1 = gsmin_SPA`` and ``gs2 = 2.0`` mol/m2/s as brackets,
    evaluates :func:`_StomataEfficiency` at both ends, and:

    - If ``check1 * check2 < 0``: uses :func:`zbrent` (``gs_solver=1``)
      or :func:`bisection` (``gs_solver=2``) to find the optimal gs
      to within ``tol = 0.001`` mol/m2/s.
    - Otherwise (low light): sets ``gs = gsmin_SPA``.

    Finally calls :func:`_CiFuncGs` to recompute photosynthesis at the
    solved gs.

    Args:
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit/shaded leaf index.
        mlcanopy_inst: Canopy container; gs, ci, and photosynthesis
            fields are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    tol: float = 0.001    # Fortran: parameter tol = 0.001_r8

    pft     = int(patch.itype[p])
    gsmin   = float(MLpftcon.gsmin_SPA[pft])
    dpai_ic = float(mlcanopy_inst.dpai_profile[p, ic])

    gs1: float = gsmin
    gs2: float = 2.0

    if dpai_ic > 0.0:                              # Fortran lines 416-426
        # Pre-extract all scalar inputs needed by _StomataEfficiencyPure
        # (one-time JAX sync cost, amortized over all solver iterations).
        _is_c3        = round(float(pftcon.c3psn[pft])) == 1
        _c3psn_val    = float(pftcon.c3psn[pft])
        _iota         = float(MLpftcon.iota_SPA[pft])
        _pref_p       = float(mlcanopy_inst.pref_forcing[p])
        _eair_ic      = float(mlcanopy_inst.eair_profile[p, ic])
        _gbv_ic       = float(mlcanopy_inst.gbv_leaf[p, ic, il])
        _lesat_ic     = float(mlcanopy_inst.leaf_esat_leaf[p, ic, il])
        _gbc_ic       = float(mlcanopy_inst.gbc_leaf[p, ic, il])
        _cair_ic      = float(mlcanopy_inst.cair_profile[p, ic])
        _vcmax_ic     = float(mlcanopy_inst.vcmax_leaf[p, ic, il])
        _je_ic        = float(mlcanopy_inst.je_leaf[p, ic, il])
        _kp_ic        = float(mlcanopy_inst.kp_leaf[p, ic, il])
        _rd_ic        = float(mlcanopy_inst.rd_leaf[p, ic, il])
        _kc_ic        = float(mlcanopy_inst.kc_leaf[p, ic, il])
        _ko_ic        = float(mlcanopy_inst.ko_leaf[p, ic, il])
        _cp_ic        = float(mlcanopy_inst.cp_leaf[p, ic, il])
        _o2ref_p      = float(mlcanopy_inst.o2ref_forcing[p])
        _apar_ic      = float(mlcanopy_inst.apar_leaf[p, ic, il])

        _scalar_kwargs = dict(
            iota=_iota, pref_p=_pref_p, eair_ic=_eair_ic,
            gbv_ic=_gbv_ic, lesat_ic=_lesat_ic,
            is_c3=_is_c3, dpai_ic=dpai_ic, gbc_ic=_gbc_ic,
            cair_ic=_cair_ic, vcmax_ic=_vcmax_ic, je_ic=_je_ic,
            kp_ic=_kp_ic, rd_ic=_rd_ic, kc_ic=_kc_ic, ko_ic=_ko_ic,
            cp_ic=_cp_ic, o2ref_p=_o2ref_p, apar_ic=_apar_ic,
            c3psn_pft_val=_c3psn_val,
        )

        # Evaluate bracket endpoints using the pure-scalar function.
        check1 = _StomataEfficiencyPure(gs1, **_scalar_kwargs)
        check2 = _StomataEfficiencyPure(gs2, **_scalar_kwargs)

        if check1 * check2 < 0.0:
            # Build closure that captures pre-extracted scalars (factory
            # pattern avoids Python loop-variable capture bug).
            def _make_func(**kw):
                def _f(gs_v):
                    return _StomataEfficiencyPure(gs_v, **kw)
                return _f
            _stomata_func = _make_func(**_scalar_kwargs)

            if gs_solver == 1:
                gs_opt = zbrent_scalar(
                    'StomataOptimization', _stomata_func, gs1, gs2, tol,
                )
            else:
                gs_opt = bisection_scalar(
                    'StomataOptimization', _stomata_func, gs1, gs2, tol,
                )
        else:
            gs_opt = gsmin

        mlcanopy_inst = mlcanopy_inst._replace(
            gs_leaf = mlcanopy_inst.gs_leaf.at[p, ic, il].set(gs_opt)
        )
    else:
        mlcanopy_inst = mlcanopy_inst._replace(
            gs_leaf = mlcanopy_inst.gs_leaf.at[p, ic, il].set(0.0)
        )

    # Final photosynthesis at solved gs — one JAX write-back call — Fortran line 428
    ci_val, mlcanopy_inst = _CiFuncGs(p, ic, il, mlcanopy_inst)
    mlcanopy_inst = mlcanopy_inst._replace(
        ci_leaf = mlcanopy_inst.ci_leaf.at[p, ic, il].set(ci_val)
    )
    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Public: leaf photosynthesis and stomatal conductance
# ---------------------------------------------------------------------------

def LeafPhotosynthesis(
    num_filter: int,
    filter_patch,
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf photosynthesis and stomatal conductance for sunlit
    (``il = isun``) or shaded (``il = isha``) leaves in each canopy
    layer.

    Mirrors Fortran subroutine ``LeafPhotosynthesis`` (public,
    lines 87-220).

    References
    ----------
    Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222.
    Bonan (2019) *Climate Change and Terrestrial Ecosystem Modeling*,
    Chapters 11-12.
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eq. (14).

    **First loop** — for each leaf layer with ``dpai > 0``:

    1. Temperature acclimation (``acclim_type``): selects ha, hd, se
       constants; for ``acclim_type = 1`` the entropy terms are
       computed from the mean air temperature ``tacclim``.
    2. Deactivation scaling at 25 °C via :func:`_fth25`.
    3. Temperature responses of Kc, Ko, Γ*, Vcmax, Jmax, Rd via
       :func:`_ft` × :func:`_fth`.
    4. C4 override of Vcmax and Rd using Q10 / sigmoid forms.
    5. ``btran = 1`` (soil moisture effect not yet implemented).
    6. Stomatal model parameters selected by ``gs_type``.
    7. Saturation vapour pressure at leaf temperature via
       :func:`MLWaterVaporMod.SatVap`; ceair constrained.
    8. Electron transport rate ``je`` via quadratic.
    9. Photosynthesis + gs solved by :func:`hybrid` (gs_type 0/1)
       or :func:`_StomataOptimization` (gs_type 2).
    10. Error checks: gs > 0, Ball-Berry/Medlyn consistency,
        diffusion consistency.

    **Second loop** — soil moisture adjustment (Fortran lines 192-212):

    - Save ``gspot = gs`` (potential unstressed conductance).
    - Apply water-stress factor ``fpsi`` from leaf water potential
      ``lwp`` via a sigmoidal function (``gspot_type = 1``) or
      ``fpsi = 1`` (``gspot_type = 0``).
    - ``gs = max(gspot * fpsi, gsmin_SPA)``.
    - Recompute photosynthesis at stressed gs via :func:`_CiFuncGs`.
    - Compute ``hs`` and ``vpd`` at the leaf surface.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; all photosynthesis and
            stomatal conductance fields are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    tol_ci:  float = 0.1     # Fortran: parameter tol = 0.1_r8
    tol_gs:  float = 0.001   # tolerance for _StomataOptimization

    c3psn_pft = pftcon.c3psn

    # ------------------------------------------------------------------
    # First loop: temperature responses + photosynthesis
    # ------------------------------------------------------------------
    for fp in range(1, num_filter + 1):            # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])
        _c3psn_val = float(c3psn_pft[pft])
        is_c3 = round(_c3psn_val) == 1
        _ncan_p = int(mlcanopy_inst.ncan_canopy[p])

        # --- Temperature acclimation — Fortran lines 135-153 ---
        # (Same for all ic within this patch, compute once.)
        if acclim_type == 0:
            vcmaxha = vcmaxha_noacclim;  jmaxha = jmaxha_noacclim
            vcmaxhd = vcmaxhd_noacclim;  jmaxhd = jmaxhd_noacclim
            vcmaxse = vcmaxse_noacclim;  jmaxse = jmaxse_noacclim
        elif acclim_type == 1:
            vcmaxha = vcmaxha_acclim;    jmaxha = jmaxha_acclim
            vcmaxhd = vcmaxhd_acclim;    jmaxhd = jmaxhd_acclim
            ta_c = min(max(float(mlcanopy_inst.tacclim_forcing[p]) - tfrz,
                           11.0), 35.0)
            vcmaxse = 668.39 - 1.07 * ta_c
            jmaxse  = 659.70 - 0.75 * ta_c
        else:
            endrun(msg=' ERROR: LeafPhotosynthesis: acclim_type not valid')
            vcmaxha = vcmaxhd = vcmaxse = jmaxha = jmaxhd = jmaxse = 0.0

        # High-temperature scaling factors — Fortran lines 155-157
        vcmaxc = _fth25(vcmaxhd, vcmaxse)
        jmaxc  = _fth25(jmaxhd, jmaxse)
        rdc    = _fth25(rdhd, rdse)

        # --- Stomatal model parameters — same for all ic ---
        if gs_type == 0:
            g0_val = float(MLpftcon.g0_MED[pft])
            g1_val = float(MLpftcon.g1_MED[pft])
        elif gs_type == 1:
            g0_val = float(MLpftcon.g0_BB[pft])
            g1_val = float(MLpftcon.g1_BB[pft])
        else:
            g0_val = -999.0;  g1_val = -999.0

        # Per-pft constants for gs_type==2
        if gs_type == 2:
            _iota_pft   = float(MLpftcon.iota_SPA[pft])
            _gsmin_pft  = float(MLpftcon.gsmin_SPA[pft])

        # --- Pre-extract constant input slices as numpy (one JAX sync each) ---
        _dpai_p    = np.asarray(mlcanopy_inst.dpai_profile[p])
        _tleaf_p   = np.asarray(mlcanopy_inst.tleaf_leaf[p, :, il])
        _vcmax25_p = np.asarray(mlcanopy_inst.vcmax25_leaf[p, :, il])
        _jmax25_p  = np.asarray(mlcanopy_inst.jmax25_leaf[p, :, il])
        _rd25_p    = np.asarray(mlcanopy_inst.rd25_leaf[p, :, il])
        _kp25_p    = np.asarray(mlcanopy_inst.kp25_leaf[p, :, il])
        _eair_p    = np.asarray(mlcanopy_inst.eair_profile[p])
        _apar_p    = np.asarray(mlcanopy_inst.apar_leaf[p, :, il])
        _gbc_p     = np.asarray(mlcanopy_inst.gbc_leaf[p, :, il])
        _gbv_p     = np.asarray(mlcanopy_inst.gbv_leaf[p, :, il])
        _cair_p    = np.asarray(mlcanopy_inst.cair_profile[p])
        _o2ref_p   = float(mlcanopy_inst.o2ref_forcing[p])
        _pref_p    = float(mlcanopy_inst.pref_forcing[p])

        # --- Numpy output arrays (accumulated over ic, then batch-written) ---
        _nmax = _ncan_p + 2
        _kc_new     = np.zeros(_nmax);  _ko_new    = np.zeros(_nmax)
        _cp_new     = np.zeros(_nmax);  _vcmax_new = np.zeros(_nmax)
        _jmax_new   = np.zeros(_nmax);  _rd_new    = np.zeros(_nmax)
        _kp_new     = np.zeros(_nmax);  _lesat_new = np.zeros(_nmax)
        _ceair_new  = np.zeros(_nmax);  _je_new    = np.zeros(_nmax)
        _gs_new     = np.zeros(_nmax);  _ci_new    = np.zeros(_nmax)
        _ac_new     = np.zeros(_nmax);  _aj_new    = np.zeros(_nmax)
        _ap_new     = np.zeros(_nmax);  _agross_new = np.zeros(_nmax)
        _anet_new   = np.zeros(_nmax);  _cs_new    = np.zeros(_nmax)

        for ic in range(1, _ncan_p + 1):

            _dpai_ic = float(_dpai_p[ic])

            if _dpai_ic > 0.0:                     # Fortran lines 159-186

                _tl = float(_tleaf_p[ic])

                # --- C3 temperature response — Fortran lines 161-166 ---
                kc_val    = kc25    * _ft(_tl, kcha)
                ko_val    = ko25    * _ft(_tl, koha)
                cp_val    = cp25    * _ft(_tl, cpha)
                vcmax_val = (float(_vcmax25_p[ic])
                             * _ft(_tl, vcmaxha) * _fth(_tl, vcmaxhd, vcmaxse, vcmaxc))
                jmax_val  = (float(_jmax25_p[ic])
                             * _ft(_tl, jmaxha)  * _fth(_tl, jmaxhd,  jmaxse,  jmaxc))
                rd_val    = (float(_rd25_p[ic])
                             * _ft(_tl, rdha) * _fth(_tl, rdhd, rdse, rdc))
                kp_val    = 0.0

                # --- C4 temperature response override — Fortran lines 168-175 ---
                if not is_c3:
                    t1 = 2.0 ** ((_tl - (tfrz + 25.0)) / 10.0)
                    t2 = 1.0 + math.exp(0.2 * ((tfrz + 15.0) - _tl))
                    t3 = 1.0 + math.exp(0.3 * (_tl - (tfrz + 40.0)))
                    t4 = 1.0 + math.exp(1.3 * (_tl - (tfrz + 55.0)))
                    vcmax_val = float(_vcmax25_p[ic]) * t1 / (t2 * t3)
                    rd_val    = float(_rd25_p[ic]) * t1 / t4
                    kp_val    = float(_kp25_p[ic]) * t1

                # btran = 1 — Fortran lines 178-179
                vcmax_val *= 1.0   # btran = 1.0

                # --- Saturation vapour pressure — Fortran lines 190-198 ---
                lesat_val, _desat = SatVap(_tl)
                _eair_ic  = float(_eair_p[ic])
                ceair_val = min(_eair_ic, lesat_val)
                if gs_type == 1:
                    ceair_val = max(ceair_val, rh_min_BB * lesat_val)

                # --- Electron transport rate — Fortran lines 200-205 ---
                _apar_ic = float(_apar_p[ic])
                qabs = 0.5 * phi_psII * _apar_ic
                bq   = -(qabs + jmax_val)
                cq   = qabs * jmax_val
                r1, r2 = quadratic(theta_j, bq, cq)
                je_val = min(r1, r2)

                # Store temperature-response values
                _kc_new[ic]    = kc_val;   _ko_new[ic]    = ko_val
                _cp_new[ic]    = cp_val;   _vcmax_new[ic] = vcmax_val
                _jmax_new[ic]  = jmax_val; _rd_new[ic]    = rd_val
                _kp_new[ic]    = kp_val;   _lesat_new[ic] = lesat_val
                _ceair_new[ic] = ceair_val; _je_new[ic]   = je_val

                _gbc_ic  = float(_gbc_p[ic])
                _gbv_ic  = float(_gbv_p[ic])
                _cair_ic = float(_cair_p[ic])

                # --- Solve for Ci / gs — Fortran lines 207-221 ---
                if gs_type in (0, 1):
                    def _make_ci_func(**kw):
                        def _f(ci_v):
                            return _CiFuncPure(ci_v, **kw)
                        return _f
                    _ci_func = _make_ci_func(
                        is_c3=is_c3, vcmax_ic=vcmax_val, je_ic=je_val,
                        kp_ic=kp_val, rd_ic=rd_val, kc_ic=kc_val,
                        ko_ic=ko_val, cp_ic=cp_val, o2ref_p=_o2ref_p,
                        cair_ic=_cair_ic, apar_ic=_apar_ic,
                        gbc_ic=_gbc_ic, gbv_ic=_gbv_ic,
                        g0_p=g0_val, g1_p=g1_val,
                        ceair_ic=ceair_val, lesat_ic=lesat_val,
                        c3psn_pft_val=_c3psn_val, dpai_ic=_dpai_ic,
                    )
                    ci0 = 0.7 * _cair_ic if is_c3 else 0.4 * _cair_ic
                    ci1 = ci0 * 0.99
                    ci_root = hybrid_scalar('LeafPhotosynthesis', _ci_func, ci0, ci1, tol_ci)

                    # Full photosynthesis at converged ci_root (pure Python, no JAX)
                    if is_c3:
                        _ac = (vcmax_val * max(ci_root - cp_val, 0.0)
                               / (ci_root + kc_val * (1.0 + _o2ref_p / ko_val)))
                        _aj = (je_val * max(ci_root - cp_val, 0.0)
                               / (4.0 * ci_root + 8.0 * cp_val))
                        _ap = 0.0
                    else:
                        _ac = vcmax_val
                        _aj = qe_c4 * _apar_ic
                        _ap = kp_val * max(ci_root, 0.0)
                    _agross = _RealizedRate(_c3psn_val, _ac, _aj, _ap)
                    _ac = max(_ac, 0.0); _aj = max(_aj, 0.0); _ap = max(_ap, 0.0)
                    _agross = max(_agross, 0.0)
                    _anet  = _agross - rd_val
                    _cs    = max(_cair_ic - _anet / _gbc_ic, 1.0)

                    # Stomatal conductance at converged ci
                    if gs_type == 1:
                        if _anet > 0.0:
                            _term = _anet / _cs
                            _bq2  = _gbv_ic - g0_val - g1_val * _term
                            _cq2  = -_gbv_ic * (g0_val + g1_val * _term * ceair_val / lesat_val)
                            r1, r2 = quadratic(1.0, _bq2, _cq2)
                            _gs = max(r1, r2)
                        else:
                            _gs = g0_val
                    else:  # gs_type == 0 (Medlyn)
                        if _anet > 0.0:
                            _vpdt = max(lesat_val - ceair_val, vpd_min_MED) * 0.001
                            _term = dh2o_to_dco2 * _anet / _cs
                            _bq2  = -(2.0 * (g0_val + _term)
                                      + (g1_val * _term) ** 2 / (_gbv_ic * _vpdt))
                            _cq2  = (g0_val * g0_val
                                     + (2.0 * g0_val
                                        + _term * (1.0 - g1_val * g1_val / _vpdt)) * _term)
                            r1, r2 = quadratic(1.0, _bq2, _cq2)
                            _gs = max(r1, r2)
                        else:
                            _gs = g0_val

                    _gs_new[ic]     = _gs;    _ci_new[ic]     = ci_root
                    _ac_new[ic]     = _ac;    _aj_new[ic]     = _aj
                    _ap_new[ic]     = _ap;    _agross_new[ic] = _agross
                    _anet_new[ic]   = _anet;  _cs_new[ic]     = _cs

                elif gs_type == 2:
                    # Inline _StomataOptimization using pre-computed local floats
                    _scalar_kwargs = dict(
                        iota=_iota_pft, pref_p=_pref_p, eair_ic=_eair_ic,
                        gbv_ic=_gbv_ic, lesat_ic=lesat_val,
                        is_c3=is_c3, dpai_ic=_dpai_ic, gbc_ic=_gbc_ic,
                        cair_ic=_cair_ic, vcmax_ic=vcmax_val, je_ic=je_val,
                        kp_ic=kp_val, rd_ic=rd_val, kc_ic=kc_val, ko_ic=ko_val,
                        cp_ic=cp_val, o2ref_p=_o2ref_p, apar_ic=_apar_ic,
                        c3psn_pft_val=_c3psn_val,
                    )
                    _check1 = _StomataEfficiencyPure(_gsmin_pft, **_scalar_kwargs)
                    _check2 = _StomataEfficiencyPure(2.0,         **_scalar_kwargs)

                    if _check1 * _check2 < 0.0:
                        def _make_sf(**kw):
                            def _sf(gs_v):
                                return _StomataEfficiencyPure(gs_v, **kw)
                            return _sf
                        _sf = _make_sf(**_scalar_kwargs)
                        if gs_solver == 1:
                            gs_opt = zbrent_scalar(
                                'StomataOptimization', _sf, _gsmin_pft, 2.0, tol_gs)
                        else:
                            gs_opt = bisection_scalar(
                                'StomataOptimization', _sf, _gsmin_pft, 2.0, tol_gs)
                    else:
                        gs_opt = _gsmin_pft

                    # Final photosynthesis at gs_opt (pure Python)
                    _cfgs_kw = {k: v for k, v in _scalar_kwargs.items()
                                if k not in ('iota', 'pref_p', 'eair_ic', 'gbv_ic', 'lesat_ic')}
                    _ci_f, _ac_f, _aj_f, _ap_f, _agross_f, _anet_f, _cs_f = (
                        _CiFuncGsPure(gs_opt, **_cfgs_kw))

                    _gs_new[ic]     = gs_opt;   _ci_new[ic]     = _ci_f
                    _ac_new[ic]     = _ac_f;    _aj_new[ic]     = _aj_f
                    _ap_new[ic]     = _ap_f;    _agross_new[ic] = _agross_f
                    _anet_new[ic]   = _anet_f;  _cs_new[ic]     = _cs_f
                    # local aliases for error check below
                    _gs  = gs_opt;  ci_root = _ci_f
                    _anet = _anet_f; _cs = _cs_f
                else:
                    endrun(msg=' ERROR: LeafPhotosynthesis: gs_type not valid')
                    _gs = 0.0; ci_root = 0.0; _anet = 0.0; _cs = 1.0

                # --- Error checks — Fortran lines 223-245 (pure Python) ---
                _gs_chk    = _gs_new[ic];   _anet_chk = _anet_new[ic]
                _cs_chk    = _cs_new[ic];   _ci_chk   = _ci_new[ic]
                _gbv_chk   = _gbv_ic;       _gbc_chk  = _gbc_ic
                _lesat_chk = lesat_val;     _ceair_chk = ceair_val
                _cair_chk  = _cair_ic

                if _gs_chk < 0.0:
                    endrun(msg=' ERROR: LeafPhotosynthesis: negative stomatal conductance')

                _hs_chk  = ((_gbv_chk * _ceair_chk + _gs_chk * _lesat_chk)
                            / ((_gbv_chk + _gs_chk) * _lesat_chk))
                _vpd_chk = (_lesat_chk - _hs_chk * _lesat_chk) * 0.001

                if gs_type == 1:
                    gs_err = (g0_val + g1_val * max(_anet_chk, 0.0)
                              * _hs_chk / _cs_chk)
                    if abs(_gs_chk - gs_err) > 1.0e-6:
                        endrun(msg=' ERROR: LeafPhotosynthesis: failed Ball-Berry error check')
                elif gs_type == 0:
                    if (_lesat_chk - _ceair_chk) > vpd_min_MED:
                        gs_err = (g0_val
                                  + dh2o_to_dco2 * (1.0 + g1_val / math.sqrt(_vpd_chk))
                                  * max(_anet_chk, 0.0) / _cs_chk)
                        if abs(_gs_chk - gs_err) > 1.0e-6:
                            endrun(msg=' ERROR: LeafPhotosynthesis: failed Medlyn error check')

                an_err = (_cair_chk - _ci_chk) / (1.0 / _gbc_chk + dh2o_to_dco2 / _gs_chk)
                if _anet_chk > 0.0 and abs(_anet_chk - an_err) > 0.01:
                    endrun(msg=' ERROR: LeafPhotosynthesis: failed diffusion error check')

            else:                                  # dpai == 0 — Fortran lines 247-257
                # rd=0, all photosynthesis outputs=0 for this layer
                _rd_new[ic]     = 0.0;  _gs_new[ic]     = 0.0
                _ci_new[ic]     = 0.0;  _ac_new[ic]     = 0.0
                _aj_new[ic]     = 0.0;  _ap_new[ic]     = 0.0
                _agross_new[ic] = 0.0;  _anet_new[ic]   = 0.0
                _cs_new[ic]     = 0.0

        # --- Batch write-back for first loop (one _replace per patch) ---
        _sl = slice(1, _ncan_p + 1)
        mlcanopy_inst = mlcanopy_inst._replace(
            kc_leaf        = mlcanopy_inst.kc_leaf.at[p, _sl, il].set(
                                 jnp.array(_kc_new[_sl])),
            ko_leaf        = mlcanopy_inst.ko_leaf.at[p, _sl, il].set(
                                 jnp.array(_ko_new[_sl])),
            cp_leaf        = mlcanopy_inst.cp_leaf.at[p, _sl, il].set(
                                 jnp.array(_cp_new[_sl])),
            vcmax_leaf     = mlcanopy_inst.vcmax_leaf.at[p, _sl, il].set(
                                 jnp.array(_vcmax_new[_sl])),
            jmax_leaf      = mlcanopy_inst.jmax_leaf.at[p, _sl, il].set(
                                 jnp.array(_jmax_new[_sl])),
            rd_leaf        = mlcanopy_inst.rd_leaf.at[p, _sl, il].set(
                                 jnp.array(_rd_new[_sl])),
            kp_leaf        = mlcanopy_inst.kp_leaf.at[p, _sl, il].set(
                                 jnp.array(_kp_new[_sl])),
            leaf_esat_leaf = mlcanopy_inst.leaf_esat_leaf.at[p, _sl, il].set(
                                 jnp.array(_lesat_new[_sl])),
            ceair_leaf     = mlcanopy_inst.ceair_leaf.at[p, _sl, il].set(
                                 jnp.array(_ceair_new[_sl])),
            je_leaf        = mlcanopy_inst.je_leaf.at[p, _sl, il].set(
                                 jnp.array(_je_new[_sl])),
            gs_leaf        = mlcanopy_inst.gs_leaf.at[p, _sl, il].set(
                                 jnp.array(_gs_new[_sl])),
            ci_leaf        = mlcanopy_inst.ci_leaf.at[p, _sl, il].set(
                                 jnp.array(_ci_new[_sl])),
            ac_leaf        = mlcanopy_inst.ac_leaf.at[p, _sl, il].set(
                                 jnp.array(_ac_new[_sl])),
            aj_leaf        = mlcanopy_inst.aj_leaf.at[p, _sl, il].set(
                                 jnp.array(_aj_new[_sl])),
            ap_leaf        = mlcanopy_inst.ap_leaf.at[p, _sl, il].set(
                                 jnp.array(_ap_new[_sl])),
            agross_leaf    = mlcanopy_inst.agross_leaf.at[p, _sl, il].set(
                                 jnp.array(_agross_new[_sl])),
            anet_leaf      = mlcanopy_inst.anet_leaf.at[p, _sl, il].set(
                                 jnp.array(_anet_new[_sl])),
            cs_leaf        = mlcanopy_inst.cs_leaf.at[p, _sl, il].set(
                                 jnp.array(_cs_new[_sl])),
            btran_soil     = mlcanopy_inst.btran_soil.at[p].set(1.0),
            g0_canopy      = mlcanopy_inst.g0_canopy.at[p].set(g0_val),
            g1_canopy      = mlcanopy_inst.g1_canopy.at[p].set(g1_val),
        )

    # ------------------------------------------------------------------
    # Second loop: soil moisture adjustment — Fortran lines 262-212
    # ------------------------------------------------------------------
    for fp in range(1, num_filter + 1):
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])
        _ncan_p   = int(mlcanopy_inst.ncan_canopy[p])
        _c3psn_val = float(c3psn_pft[pft])
        is_c3     = round(_c3psn_val) == 1
        _gsmin_pft2 = float(MLpftcon.gsmin_SPA[pft])

        # Pre-extract slices for second loop (one JAX sync each)
        _gs_p2      = np.asarray(mlcanopy_inst.gs_leaf[p, :, il])
        _dpai_p2    = np.asarray(mlcanopy_inst.dpai_profile[p])
        _gbv_p2     = np.asarray(mlcanopy_inst.gbv_leaf[p, :, il])
        _eair_p2    = np.asarray(mlcanopy_inst.eair_profile[p])
        _lesat_p2   = np.asarray(mlcanopy_inst.leaf_esat_leaf[p, :, il])
        _gbc_p2     = np.asarray(mlcanopy_inst.gbc_leaf[p, :, il])
        _vcmax_p2   = np.asarray(mlcanopy_inst.vcmax_leaf[p, :, il])
        _je_p2      = np.asarray(mlcanopy_inst.je_leaf[p, :, il])
        _kp_p2      = np.asarray(mlcanopy_inst.kp_leaf[p, :, il])
        _rd_p2      = np.asarray(mlcanopy_inst.rd_leaf[p, :, il])
        _kc_p2      = np.asarray(mlcanopy_inst.kc_leaf[p, :, il])
        _ko_p2      = np.asarray(mlcanopy_inst.ko_leaf[p, :, il])
        _cp_p2      = np.asarray(mlcanopy_inst.cp_leaf[p, :, il])
        _o2ref_p2   = float(mlcanopy_inst.o2ref_forcing[p])
        _cair_p2    = np.asarray(mlcanopy_inst.cair_profile[p])
        _apar_p2    = np.asarray(mlcanopy_inst.apar_leaf[p, :, il])
        _ceair_p2   = np.asarray(mlcanopy_inst.ceair_leaf[p, :, il])

        if gspot_type == 1:
            _lwp_p2     = np.asarray(mlcanopy_inst.lwp_leaf[p, :, il])
            _psi50_pft2 = float(MLpftcon.psi50_gs[pft])
            _shape_pft2 = float(MLpftcon.shape_gs[pft])

        # Output arrays for second loop
        _nmax2      = _ncan_p + 2
        _gspot_new2 = np.zeros(_nmax2)
        _gs2_new    = np.zeros(_nmax2)
        _ci2_new    = np.zeros(_nmax2)
        _ac2_new    = np.zeros(_nmax2);  _aj2_new    = np.zeros(_nmax2)
        _ap2_new    = np.zeros(_nmax2);  _agross2_new = np.zeros(_nmax2)
        _anet2_new  = np.zeros(_nmax2);  _cs2_new    = np.zeros(_nmax2)
        _hs_new     = np.zeros(_nmax2);  _vpd_new    = np.zeros(_nmax2)

        for ic in range(1, _ncan_p + 1):

            gs_ic = float(_gs_p2[ic])
            _gspot_new2[ic] = gs_ic                # save potential (unstressed) gs

            _dpai_ic2 = float(_dpai_p2[ic])

            if _dpai_ic2 > 0.0:                    # Fortran lines 271-284

                # Water-stress factor — Fortran lines 273-277
                if gspot_type == 0:
                    fpsi = 1.0
                elif gspot_type == 1:
                    lwp_ic  = float(_lwp_p2[ic])
                    fpsi    = 1.0 / (1.0 + (lwp_ic / _psi50_pft2) ** _shape_pft2)
                else:
                    fpsi = 1.0

                gs_new2 = max(gs_ic * fpsi, _gsmin_pft2)
                _gs2_new[ic] = gs_new2

                # Recalculate photosynthesis via pure scalar (no JAX reads/writes)
                _ci_f2, _ac_f2, _aj_f2, _ap_f2, _agross_f2, _anet_f2, _cs_f2 = (
                    _CiFuncGsPure(
                        gs_new2,
                        is_c3=is_c3, dpai_ic=_dpai_ic2,
                        gbc_ic=float(_gbc_p2[ic]), cair_ic=float(_cair_p2[ic]),
                        vcmax_ic=float(_vcmax_p2[ic]), je_ic=float(_je_p2[ic]),
                        kp_ic=float(_kp_p2[ic]),    rd_ic=float(_rd_p2[ic]),
                        kc_ic=float(_kc_p2[ic]),    ko_ic=float(_ko_p2[ic]),
                        cp_ic=float(_cp_p2[ic]),    o2ref_p=_o2ref_p2,
                        apar_ic=float(_apar_p2[ic]), c3psn_pft_val=_c3psn_val,
                    )
                )
                _ci2_new[ic]     = _ci_f2;   _ac2_new[ic]     = _ac_f2
                _aj2_new[ic]     = _aj_f2;   _ap2_new[ic]     = _ap_f2
                _agross2_new[ic] = _agross_f2; _anet2_new[ic] = _anet_f2
                _cs2_new[ic]     = _cs_f2

                # hs and vpd at leaf surface — Fortran lines 282-284
                _gbv_ic2  = float(_gbv_p2[ic])
                _eair_ic2 = float(_eair_p2[ic])
                _lesat_ic2 = float(_lesat_p2[ic])
                hs_val  = ((_gbv_ic2 * _eair_ic2 + gs_new2 * _lesat_ic2)
                           / ((_gbv_ic2 + gs_new2) * _lesat_ic2))
                vpd_val = max(_lesat_ic2 - hs_val * _lesat_ic2, 0.1)
                _hs_new[ic]  = hs_val
                _vpd_new[ic] = vpd_val

            else:                                  # Fortran lines 286-288
                _gs2_new[ic]  = gs_ic              # unchanged
                _hs_new[ic]   = 0.0
                _vpd_new[ic]  = 0.0

        # --- Batch write-back for second loop ---
        _sl2 = slice(1, _ncan_p + 1)
        mlcanopy_inst = mlcanopy_inst._replace(
            gspot_leaf  = mlcanopy_inst.gspot_leaf.at[p, _sl2, il].set(
                              jnp.array(_gspot_new2[_sl2])),
            gs_leaf     = mlcanopy_inst.gs_leaf.at[p, _sl2, il].set(
                              jnp.array(_gs2_new[_sl2])),
            ci_leaf     = mlcanopy_inst.ci_leaf.at[p, _sl2, il].set(
                              jnp.array(_ci2_new[_sl2])),
            ac_leaf     = mlcanopy_inst.ac_leaf.at[p, _sl2, il].set(
                              jnp.array(_ac2_new[_sl2])),
            aj_leaf     = mlcanopy_inst.aj_leaf.at[p, _sl2, il].set(
                              jnp.array(_aj2_new[_sl2])),
            ap_leaf     = mlcanopy_inst.ap_leaf.at[p, _sl2, il].set(
                              jnp.array(_ap2_new[_sl2])),
            agross_leaf = mlcanopy_inst.agross_leaf.at[p, _sl2, il].set(
                              jnp.array(_agross2_new[_sl2])),
            anet_leaf   = mlcanopy_inst.anet_leaf.at[p, _sl2, il].set(
                              jnp.array(_anet2_new[_sl2])),
            cs_leaf     = mlcanopy_inst.cs_leaf.at[p, _sl2, il].set(
                              jnp.array(_cs2_new[_sl2])),
            hs_leaf     = mlcanopy_inst.hs_leaf.at[p, _sl2, il].set(
                              jnp.array(_hs_new[_sl2])),
            vpd_leaf    = mlcanopy_inst.vpd_leaf.at[p, _sl2, il].set(
                              jnp.array(_vpd_new[_sl2])),
        )

    return mlcanopy_inst