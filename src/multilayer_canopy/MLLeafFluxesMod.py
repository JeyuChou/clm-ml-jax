"""
JAX translation of MLLeafFluxesMod Fortran module.

Leaf temperature and energy fluxes for the multilayer canopy model.

Original Fortran module: MLLeafFluxesMod
Fortran lines 1-115
"""

from __future__ import annotations

from clm_src_main.abortutils import endrun                       # noqa: F401
from multilayer_canopy.MLclm_varctl import dtime_ml                  # noqa: F401
from multilayer_canopy.MLWaterVaporMod import SatVap, LatVap         # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type       # noqa: F401


def LeafFluxes(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf temperature and energy fluxes.

    Mirrors Fortran subroutine ``LeafFluxes`` (lines 22-115).

    Reference: Bonan et al. (2018) *Geosci. Model Dev.*, 11, 1467-1496,
    doi:10.5194/gmd-11-1467-2018, eqs. (10)-(13).

    The leaf temperature is obtained by linearising the energy balance
    around ``tleaf_bef`` (the previous timestep's leaf temperature):

    .. code-block:: none

        gleaf = gs * gbv / (gs + gbv)                   [transpiration conductance]
        gw    = gleaf * fdry + gbv * fwet               [total conductance]

        num1  = 2 * cpair * gbh
        num2  = lambda * gw
        num3  = rnleaf - lambda*gw*(qsat - dqsat*tleaf_bef)
                + (cpleaf/dtime) * tleaf_bef
        den   = cpleaf/dtime + num1 + num2*dqsat
        tleaf = (num1*tair + num2*(eair/pref) + num3) / den

    Derived fluxes (Fortran lines 87-100):

    .. code-block:: none

        stleaf = (tleaf - tleaf_bef) * cpleaf / dtime          [W/m2 leaf]
        shleaf = 2 * cpair * (tleaf - tair) * gbh              [W/m2 leaf]
        num1   = qsat + dqsat*(tleaf - tleaf_bef) - eair/pref
        trleaf = gleaf * fdry * num1                            [mol H2O/m2/s]
        evleaf = gbv   * fwet * num1                            [mol H2O/m2/s]
        lhleaf = (trleaf + evleaf) * lambda                     [W/m2 leaf]

    Energy balance closure is checked to within 1e-3 W/m2.

    Layers with ``dpai == 0`` have all fluxes set to zero and
    ``tleaf = tair(p, ic)``.

    Args:
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; ``tleaf_leaf``,
            ``stleaf_leaf``, ``shleaf_leaf``, ``lhleaf_leaf``,
            ``evleaf_leaf``, and ``trleaf_leaf`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dtime: float = float(dtime_ml)    # Multilayer canopy timestep (s)

    # Latent heat of vaporization — Fortran line 65
    lam = LatVap(float(mlcanopy_inst.tref_forcing[p]))    # J/mol

    dpai_ic = float(mlcanopy_inst.dpai_profile[p, ic])

    if dpai_ic > 0.0:                                      # Fortran lines 67-105

        # Unpack scalars
        pref_p       = float(mlcanopy_inst.pref_forcing[p])
        cpair_p      = float(mlcanopy_inst.cpair_forcing[p])
        tair_ic      = float(mlcanopy_inst.tair_profile[p, ic])
        eair_ic      = float(mlcanopy_inst.eair_profile[p, ic])
        cpleaf_ic    = float(mlcanopy_inst.cpleaf_profile[p, ic])
        fwet_ic      = float(mlcanopy_inst.fwet_profile[p, ic])
        fdry_ic      = float(mlcanopy_inst.fdry_profile[p, ic])
        gbh_ic       = float(mlcanopy_inst.gbh_leaf[p, ic, il])
        gbv_ic       = float(mlcanopy_inst.gbv_leaf[p, ic, il])
        gs_ic        = float(mlcanopy_inst.gs_leaf[p, ic, il])
        rnleaf_ic    = float(mlcanopy_inst.rnleaf_leaf[p, ic, il])
        tleaf_bef_ic = float(mlcanopy_inst.tleaf_bef_leaf[p, ic, il])

        # Saturation vapour pressure at tleaf_bef — Fortran lines 69-70
        esat, desat = SatVap(tleaf_bef_ic)
        qsat  = esat  / pref_p                             # mol/mol
        dqsat = desat / pref_p                             # mol/mol/K

        # Leaf transpiration conductance — Fortran line 72
        gleaf = gs_ic * gbv_ic / (gs_ic + gbv_ic)

        # Total conductance (transpiration + evaporation) — Fortran line 74
        gw = gleaf * fdry_ic + gbv_ic * fwet_ic

        # Linearised leaf temperature — Fortran lines 76-81
        num1   = 2.0 * cpair_p * gbh_ic
        num2   = lam * gw
        num3   = (rnleaf_ic
                  - lam * gw * (qsat - dqsat * tleaf_bef_ic)
                  + cpleaf_ic / dtime * tleaf_bef_ic)
        den    = cpleaf_ic / dtime + num1 + num2 * dqsat
        tleaf_val = (num1 * tair_ic + num2 * (eair_ic / pref_p) + num3) / den

        # Storage heat flux — Fortran line 83
        stleaf_val = (tleaf_val - tleaf_bef_ic) * cpleaf_ic / dtime

        # Sensible heat flux — Fortran line 85
        shleaf_val = 2.0 * cpair_p * (tleaf_val - tair_ic) * gbh_ic

        # Transpiration and evaporation fluxes (mol H2O/m2/s) — Fortran lines 87-89
        num1_flux   = qsat + dqsat * (tleaf_val - tleaf_bef_ic) - eair_ic / pref_p
        trleaf_val  = gleaf * fdry_ic * num1_flux
        evleaf_val  = gbv_ic * fwet_ic * num1_flux

        # Latent heat flux — Fortran line 91
        lhleaf_val  = (trleaf_val + evleaf_val) * lam

        # Energy balance error check — Fortran lines 93-96
        err = rnleaf_ic - shleaf_val - lhleaf_val - stleaf_val
        if abs(err) > 1.0e-3:
            endrun(msg=' ERROR: LeafFluxes: energy balance error')

    else:                                                  # Fortran lines 98-104
        tleaf_val  = float(mlcanopy_inst.tair_profile[p, ic])
        stleaf_val = 0.0
        shleaf_val = 0.0
        lhleaf_val = 0.0
        evleaf_val = 0.0
        trleaf_val = 0.0

    return mlcanopy_inst._replace(
        tleaf_leaf  = mlcanopy_inst.tleaf_leaf.at[p, ic, il].set(tleaf_val),
        stleaf_leaf = mlcanopy_inst.stleaf_leaf.at[p, ic, il].set(stleaf_val),
        shleaf_leaf = mlcanopy_inst.shleaf_leaf.at[p, ic, il].set(shleaf_val),
        lhleaf_leaf = mlcanopy_inst.lhleaf_leaf.at[p, ic, il].set(lhleaf_val),
        evleaf_leaf = mlcanopy_inst.evleaf_leaf.at[p, ic, il].set(evleaf_val),
        trleaf_leaf = mlcanopy_inst.trleaf_leaf.at[p, ic, il].set(trleaf_val),
    )