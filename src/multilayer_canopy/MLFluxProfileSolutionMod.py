"""
JAX translation of MLFluxProfileSolutionMod Fortran module.

Source/sink fluxes for leaves and soil, and concentration profiles
for the multilayer canopy model (CLM/CTSM).

Original Fortran module: MLFluxProfileSolutionMod
Fortran lines 1-430
"""

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import Array

from clm_src_main.abortutils import endrun                                          # noqa: F401
from clm_src_main.clm_varctl import iulog                                          # noqa: F401
from multilayer_canopy.MLclm_varctl import flux_profile_type, dtime_ml                 # noqa: F401
from multilayer_canopy.MLclm_varpar import isun, isha, nlevmlcan, nleaf                 # noqa: F401
from multilayer_canopy.MLWaterVaporMod import SatVap, LatVap                            # noqa: F401
from multilayer_canopy.MLMathToolsMod import tridiag_2eq                                # noqa: F401
from multilayer_canopy.MLLeafFluxesMod import LeafFluxes                                # noqa: F401
from multilayer_canopy.MLSoilFluxesMod import SoilFluxes                                # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type                          # noqa: F401


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def FluxProfileSolution(
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Source/sink fluxes for leaves and soil, and concentration profiles.

    Mirrors Fortran subroutine ``FluxProfileSolution`` (lines 27-84).
    Dispatches to ``WellMixed`` or ``ImplicitFluxProfileSolution`` based
    on ``flux_profile_type``.

    Args:
        num_filter: Number of patches in the filter.
            Fortran: ``integer, intent(in) :: num_filter``.
        filter: 1-D array of patch indices to process.
            Fortran: ``integer, intent(in) :: filter(:)``.
        mlcanopy_inst: Multilayer canopy state container (immutable
            NamedTuple); updated fields are returned in a new instance.

    Returns:
        Updated :class:`mlcanopy_type` after flux/profile calculation.

    Raises:
        RuntimeError: If ``flux_profile_type`` is 0 or -1 (WellMixed
            legacy path, not supported) or unrecognised.
    """
    co2ref = mlcanopy_inst.co2ref_forcing     # Atmospheric CO2 at ref height (umol/mol)
    ncan   = mlcanopy_inst.ncan_canopy        # Number of aboveground layers
    cair   = mlcanopy_inst.cair_profile       # Canopy layer atmospheric CO2 (umol/mol)

    if flux_profile_type in (0, -1):
        # Legacy well-mixed path — Fortran lines 52-58
        mlcanopy_inst = WellMixed(num_filter, filter, mlcanopy_inst)
        endrun(msg=' ERROR: WellMixed option not valid')

    elif flux_profile_type == 1:
        # Implicit flux-profile solution — Fortran lines 62-74
        for fp in range(num_filter):
            p = int(filter[fp])
            mlcanopy_inst = ImplicitFluxProfileSolution(p, mlcanopy_inst)

            # No profile for CO2: set every layer to reference value
            # Fortran lines 68-70
            n = int(ncan[p])
            cair = cair.at[p, :n].set(co2ref[p])
            mlcanopy_inst = mlcanopy_inst._replace(cair_profile=cair)

    else:
        endrun(msg=' ERROR: FluxProfileSolution: flux_profile_type not valid')

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: implicit flux-profile solution
# ---------------------------------------------------------------------------

def ImplicitFluxProfileSolution(
    p: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Implicit solution for source/sink fluxes and concentration profiles.

    Computes leaf and soil fluxes together with canopy air temperature
    and water-vapour profiles via a coupled implicit tridiagonal solve.
    Boundary conditions are the above-canopy scalar values at reference
    height and the temperature of the first soil layer.

    Mirrors Fortran subroutine ``ImplicitFluxProfileSolution``
    (lines 86-310).

    Reference:
        Bonan et al. (2018) Geosci. Model Dev., 11, 1467-1496,
        doi:10.5194/gmd-11-1467-2018, eqs. (10)-(17), (A1)-(A9),
        (S10)-(S31).

    Args:
        p: Patch index for the CLM g/l/c/p hierarchy.
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Updated :class:`mlcanopy_type` with leaf, soil, and canopy-air
        fluxes and profiles populated.
    """
    # ------------------------------------------------------------------
    # Unpack inputs from mlcanopy_inst  (Fortran associate block)
    # ------------------------------------------------------------------
    tair   = mlcanopy_inst.tair_profile          # Canopy layer air temperature (K)
    eair   = mlcanopy_inst.eair_profile          # Canopy layer vapor pressure (Pa)
    shair  = mlcanopy_inst.shair_profile         # Canopy layer air SH flux (W/m2)
    etair  = mlcanopy_inst.etair_profile         # Canopy layer air ET flux (mol H2O/m2/s)
    stair  = mlcanopy_inst.stair_profile         # Canopy layer air storage flux (W/m2)

    # ------------------------------------------------------------------
    # Pre-extract all patch-p slices as numpy to avoid per-element JAX syncs
    # ------------------------------------------------------------------
    _pref_p    = float(mlcanopy_inst.pref_forcing[p])
    _rhomol_p  = float(mlcanopy_inst.rhomol_forcing[p])
    _cpair_p   = float(mlcanopy_inst.cpair_forcing[p])
    _tref_p    = float(mlcanopy_inst.tref_forcing[p])
    _thref_p   = float(mlcanopy_inst.thref_forcing[p])
    _eref_p    = float(mlcanopy_inst.eref_forcing[p])
    _tg_bef_p  = float(mlcanopy_inst.tg_bef_soil[p])
    _rhg_p     = float(mlcanopy_inst.rhg_soil[p])
    _rnsoi_p   = float(mlcanopy_inst.rnsoi_soil[p])
    _soilres_p = float(mlcanopy_inst.soilres_soil[p])
    _soil_t_p  = float(mlcanopy_inst.soil_t_soil[p])
    _soil_dz_p = float(mlcanopy_inst.soil_dz_soil[p])
    _soil_tk_p = float(mlcanopy_inst.soil_tk_soil[p])
    _gac0_p    = float(mlcanopy_inst.gac0_soil[p])
    n          = int(mlcanopy_inst.ncan_canopy[p])

    # 1-D and 2-D slices (single sync per array, then pure numpy in loops)
    _dz_p        = np.asarray(mlcanopy_inst.dz_profile[p])
    _dpai_p      = np.asarray(mlcanopy_inst.dpai_profile[p])
    _fwet_p      = np.asarray(mlcanopy_inst.fwet_profile[p])
    _fdry_p      = np.asarray(mlcanopy_inst.fdry_profile[p])
    _fracsun_p   = np.asarray(mlcanopy_inst.fracsun_profile[p])
    _cpleaf_p    = np.asarray(mlcanopy_inst.cpleaf_profile[p])
    _gac_p       = np.asarray(mlcanopy_inst.gac_profile[p])
    _tair_bef_p  = np.asarray(mlcanopy_inst.tair_bef_profile[p])
    _eair_bef_p  = np.asarray(mlcanopy_inst.eair_bef_profile[p])
    _gbh_p       = np.asarray(mlcanopy_inst.gbh_leaf[p])
    _gbv_p       = np.asarray(mlcanopy_inst.gbv_leaf[p])
    _gs_p        = np.asarray(mlcanopy_inst.gs_leaf[p])
    _rnleaf_p    = np.asarray(mlcanopy_inst.rnleaf_leaf[p])
    _tleaf_bef_p = np.asarray(mlcanopy_inst.tleaf_bef_leaf[p])

    # ------------------------------------------------------------------
    # Timestep and latent heat — Fortran lines 155-158
    # ------------------------------------------------------------------
    dtime   = dtime_ml
    lambda_ = LatVap(_tref_p)

    # ------------------------------------------------------------------
    # Ground temperature coefficients alpha0, beta0, delta0
    # Fortran lines 168-177
    # ------------------------------------------------------------------
    esat, desat = SatVap(_tg_bef_p)
    qsat0  = esat  / _pref_p
    dqsat0 = desat / _pref_p

    gsw  = (1.0 / _soilres_p) * _rhomol_p
    gs0  = _gac0_p * gsw / (_gac0_p + gsw)

    c02  = _soil_tk_p / _soil_dz_p
    c01  = -c02 * _soil_t_p

    den    = _cpair_p * _gac0_p + lambda_ * _rhg_p * gs0 * dqsat0 + c02
    alpha0 = _cpair_p * _gac0_p / den
    beta0  = lambda_ * gs0 / den
    delta0 = (_rnsoi_p - lambda_ * _rhg_p * gs0 * (qsat0 - dqsat0 * _tg_bef_p) - c01) / den

    # ------------------------------------------------------------------
    # Leaf temperature coefficients — numpy local arrays (no JAX syncs)
    # Bonan et al. (2018) eqs. (10)-(13), (A1)-(A5); Fortran lines 187-247
    # ------------------------------------------------------------------
    gleaf_sh     = np.zeros((nlevmlcan + 1, nleaf + 1))
    gleaf_et     = np.zeros((nlevmlcan + 1, nleaf + 1))
    heatcap      = np.zeros((nlevmlcan + 1, nleaf + 1))
    avail_energy = np.zeros((nlevmlcan + 1, nleaf + 1))
    dqsat_arr    = np.zeros((nlevmlcan + 1, nleaf + 1))
    qsat_term    = np.zeros((nlevmlcan + 1, nleaf + 1))
    alpha        = np.zeros((nlevmlcan + 1, nleaf + 1))
    beta         = np.zeros((nlevmlcan + 1, nleaf + 1))
    delta        = np.zeros((nlevmlcan + 1, nleaf + 1))

    for ic in range(1, n + 1):
        if _dpai_p[ic] > 0.0:
            for il in [isun, isha]:
                gleaf_sh[ic, il] = 2.0 * _gbh_p[ic, il]
                _gs_il  = _gs_p[ic, il]
                _gbv_il = _gbv_p[ic, il]
                gleaf_et[ic, il] = (
                    _gs_il * _gbv_il / (_gs_il + _gbv_il) * _fdry_p[ic]
                    + _gbv_il * _fwet_p[ic]
                )
                heatcap[ic, il]      = _cpleaf_p[ic]
                avail_energy[ic, il] = _rnleaf_p[ic, il]

                esat_l, desat_l = SatVap(_tleaf_bef_p[ic, il])
                qsat_l  = esat_l  / _pref_p
                dqsat_l = desat_l / _pref_p
                dqsat_arr[ic, il] = dqsat_l
                qsat_term[ic, il] = qsat_l - dqsat_l * _tleaf_bef_p[ic, il]

                den_l = (
                    heatcap[ic, il] / dtime
                    + gleaf_sh[ic, il] * _cpair_p
                    + gleaf_et[ic, il] * lambda_ * dqsat_l
                )
                alpha[ic, il] = gleaf_sh[ic, il] * _cpair_p / den_l
                beta[ic, il]  = gleaf_et[ic, il] * lambda_ / den_l
                delta[ic, il] = (
                    avail_energy[ic, il] / den_l
                    - lambda_ * gleaf_et[ic, il] * qsat_term[ic, il] / den_l
                    + heatcap[ic, il] / dtime * _tleaf_bef_p[ic, il] / den_l
                )

                pai = (_dpai_p[ic] * _fracsun_p[ic] if il == isun
                       else _dpai_p[ic] * (1.0 - _fracsun_p[ic]))
                gleaf_sh[ic, il]     *= pai
                gleaf_et[ic, il]     *= pai
                heatcap[ic, il]      *= pai
                avail_energy[ic, il] *= pai

    # ------------------------------------------------------------------
    # Tridiagonal coefficients — numpy local arrays
    # Bonan et al. (2018) eqs. (16)-(17), (S10)-(S31); Fortran lines 258-316
    # ------------------------------------------------------------------
    rho_dz_over_dt = np.zeros(nlevmlcan + 1)
    a1  = np.zeros(nlevmlcan + 1)
    b11 = np.zeros(nlevmlcan + 1)
    b12 = np.zeros(nlevmlcan + 1)
    c1  = np.zeros(nlevmlcan + 1)
    d1  = np.zeros(nlevmlcan + 1)
    a2  = np.zeros(nlevmlcan + 1)
    b21 = np.zeros(nlevmlcan + 1)
    b22 = np.zeros(nlevmlcan + 1)
    c2  = np.zeros(nlevmlcan + 1)
    d2  = np.zeros(nlevmlcan + 1)

    for ic in range(1, n + 1):
        rho_dz_over_dt[ic] = _rhomol_p * _dz_p[ic] / dtime

        gac_below_sh = _gac0_p if ic == 1 else _gac_p[ic - 1]
        gac_below_et = gs0     if ic == 1 else _gac_p[ic - 1]
        gac_ic       = _gac_p[ic]

        a1[ic]  = -gac_below_sh
        b11[ic] = (rho_dz_over_dt[ic] + gac_below_sh + gac_ic
                   + gleaf_sh[ic, isun] * (1.0 - alpha[ic, isun])
                   + gleaf_sh[ic, isha] * (1.0 - alpha[ic, isha]))
        b12[ic] = (-gleaf_sh[ic, isun] * beta[ic, isun]
                   - gleaf_sh[ic, isha] * beta[ic, isha])
        c1[ic]  = -gac_ic
        d1[ic]  = (rho_dz_over_dt[ic] * _tair_bef_p[ic]
                   + gleaf_sh[ic, isun] * delta[ic, isun]
                   + gleaf_sh[ic, isha] * delta[ic, isha])

        if ic == n:
            c1[ic]  = 0.0
            d1[ic] += gac_ic * _thref_p
        if ic == 1:
            a1[ic]   = 0.0
            b11[ic] += -_gac0_p * alpha0
            b12[ic] += -_gac0_p * beta0
            d1[ic]  +=  _gac0_p * delta0

        a2[ic]  = -gac_below_et
        b21[ic] = (-gleaf_et[ic, isun] * dqsat_arr[ic, isun] * alpha[ic, isun]
                   - gleaf_et[ic, isha] * dqsat_arr[ic, isha] * alpha[ic, isha])
        b22[ic] = (rho_dz_over_dt[ic] + gac_below_et + gac_ic
                   + gleaf_et[ic, isun] * (1.0 - dqsat_arr[ic, isun] * beta[ic, isun])
                   + gleaf_et[ic, isha] * (1.0 - dqsat_arr[ic, isha] * beta[ic, isha]))
        c2[ic]  = -gac_ic
        d2[ic]  = (rho_dz_over_dt[ic] * (_eair_bef_p[ic] / _pref_p)
                   + gleaf_et[ic, isun] * (dqsat_arr[ic, isun] * delta[ic, isun] + qsat_term[ic, isun])
                   + gleaf_et[ic, isha] * (dqsat_arr[ic, isha] * delta[ic, isha] + qsat_term[ic, isha]))

        if ic == n:
            c2[ic]  = 0.0
            d2[ic] += gac_ic * (_eref_p / _pref_p)
        if ic == 1:
            a2[ic]   = 0.0
            b21[ic] += -gs0 * _rhg_p * dqsat0 * alpha0
            b22[ic] += -gs0 * _rhg_p * dqsat0 * beta0
            d2[ic]  +=  gs0 * _rhg_p * (qsat0 + dqsat0 * (delta0 - _tg_bef_p))

    # ------------------------------------------------------------------
    # Tridiagonal solve for tair and eair (mol/mol) — Fortran lines 327-329
    # ------------------------------------------------------------------
    tair_p, eair_p = tridiag_2eq(
        a1[1:n+1], b11[1:n+1], b12[1:n+1], c1[1:n+1], d1[1:n+1],
        a2[1:n+1], b21[1:n+1], b22[1:n+1], c2[1:n+1], d2[1:n+1],
        n,
    )
    tair = tair.at[p, 1:n+1].set(jnp.array(tair_p, dtype=jnp.float64))
    eair = eair.at[p, 1:n+1].set(jnp.array(eair_p, dtype=jnp.float64))

    # ------------------------------------------------------------------
    # Soil surface temperature and vapor pressure — Fortran lines 332-333
    # ------------------------------------------------------------------
    # Convert to numpy for subsequent pure-Python arithmetic
    _tair_solved = np.asarray(tair[p])
    _eair_solved = np.asarray(eair[p])

    t0 = alpha0 * _tair_solved[1] + beta0 * _eair_solved[1] + delta0
    e0 = _rhg_p * (qsat0 + dqsat0 * (t0 - _tg_bef_p))

    # ------------------------------------------------------------------
    # Leaf temperature from implicit solution — Fortran lines 336-340
    # ------------------------------------------------------------------
    tleaf_implic = np.zeros((nlevmlcan + 1, nleaf + 1))
    for ic in range(1, n + 1):
        tleaf_implic[ic, isun] = (alpha[ic, isun] * _tair_solved[ic]
                                  + beta[ic, isun] * _eair_solved[ic]
                                  + delta[ic, isun])
        tleaf_implic[ic, isha] = (alpha[ic, isha] * _tair_solved[ic]
                                  + beta[ic, isha] * _eair_solved[ic]
                                  + delta[ic, isha])

    # ------------------------------------------------------------------
    # Convert water vapor from mol/mol to Pa — Fortran lines 343-346
    # ------------------------------------------------------------------
    eair = eair.at[p, 1:n+1].set(eair[p, 1:n+1] * _pref_p)
    e0   = e0 * _pref_p

    mlcanopy_inst = mlcanopy_inst._replace(
        tair_profile=tair,
        eair_profile=eair,
    )

    # ------------------------------------------------------------------
    # Leaf fluxes (per unit leaf area) — Fortran lines 352-354
    # Inlined from LeafFluxes; all inputs from pre-extracted numpy arrays
    # ------------------------------------------------------------------
    # eair in Pa (as stored after the _replace at lines 342-345)
    _eair_lf_p = _eair_solved.copy()
    _eair_lf_p[1:n + 1] = _eair_solved[1:n + 1] * _pref_p

    _tleaf_new  = np.zeros((nlevmlcan + 1, nleaf + 1))
    _stleaf_new = np.zeros((nlevmlcan + 1, nleaf + 1))
    _shleaf_new = np.zeros((nlevmlcan + 1, nleaf + 1))
    _lhleaf_new = np.zeros((nlevmlcan + 1, nleaf + 1))
    _evleaf_new = np.zeros((nlevmlcan + 1, nleaf + 1))
    _trleaf_new = np.zeros((nlevmlcan + 1, nleaf + 1))

    for ic in range(1, n + 1):
        tair_ic = float(_tair_solved[ic])
        eair_ic = float(_eair_lf_p[ic])
        for il in [isun, isha]:
            if _dpai_p[ic] > 0.0:
                cpleaf_ic    = float(_cpleaf_p[ic])
                fwet_ic      = float(_fwet_p[ic])
                fdry_ic      = float(_fdry_p[ic])
                gbh_ic       = float(_gbh_p[ic, il])
                gbv_ic       = float(_gbv_p[ic, il])
                gs_ic        = float(_gs_p[ic, il])
                rnleaf_ic    = float(_rnleaf_p[ic, il])
                tleaf_bef_ic = float(_tleaf_bef_p[ic, il])

                esat_l, desat_l = SatVap(tleaf_bef_ic)
                qsat_lf  = esat_l  / _pref_p
                dqsat_lf = desat_l / _pref_p

                gleaf = gs_ic * gbv_ic / (gs_ic + gbv_ic)
                gw    = gleaf * fdry_ic + gbv_ic * fwet_ic

                num1  = 2.0 * _cpair_p * gbh_ic
                num2  = lambda_ * gw
                num3  = (rnleaf_ic
                         - lambda_ * gw * (qsat_lf - dqsat_lf * tleaf_bef_ic)
                         + cpleaf_ic / dtime * tleaf_bef_ic)
                den   = cpleaf_ic / dtime + num1 + num2 * dqsat_lf
                tleaf_val = (num1 * tair_ic + num2 * (eair_ic / _pref_p) + num3) / den

                stleaf_val = (tleaf_val - tleaf_bef_ic) * cpleaf_ic / dtime
                shleaf_val = 2.0 * _cpair_p * (tleaf_val - tair_ic) * gbh_ic

                num1_flux  = qsat_lf + dqsat_lf * (tleaf_val - tleaf_bef_ic) - eair_ic / _pref_p
                trleaf_val = gleaf * fdry_ic * num1_flux
                evleaf_val = gbv_ic * fwet_ic * num1_flux
                lhleaf_val = (trleaf_val + evleaf_val) * lambda_

                err = rnleaf_ic - shleaf_val - lhleaf_val - stleaf_val
                if abs(err) > 1.0e-3:
                    endrun(msg=' ERROR: LeafFluxes: energy balance error')
            else:
                tleaf_val  = tair_ic
                stleaf_val = 0.0
                shleaf_val = 0.0
                lhleaf_val = 0.0
                evleaf_val = 0.0
                trleaf_val = 0.0

            _tleaf_new[ic, il]  = tleaf_val
            _stleaf_new[ic, il] = stleaf_val
            _shleaf_new[ic, il] = shleaf_val
            _lhleaf_new[ic, il] = lhleaf_val
            _evleaf_new[ic, il] = evleaf_val
            _trleaf_new[ic, il] = trleaf_val

    # Batch write-back: 6 bulk operations instead of 120 per-element writes
    _sl_lf = slice(1, n + 1)
    mlcanopy_inst = mlcanopy_inst._replace(
        tleaf_leaf  = mlcanopy_inst.tleaf_leaf.at[p, _sl_lf, :].set(
                          jnp.array(_tleaf_new[_sl_lf, :])),
        stleaf_leaf = mlcanopy_inst.stleaf_leaf.at[p, _sl_lf, :].set(
                          jnp.array(_stleaf_new[_sl_lf, :])),
        shleaf_leaf = mlcanopy_inst.shleaf_leaf.at[p, _sl_lf, :].set(
                          jnp.array(_shleaf_new[_sl_lf, :])),
        lhleaf_leaf = mlcanopy_inst.lhleaf_leaf.at[p, _sl_lf, :].set(
                          jnp.array(_lhleaf_new[_sl_lf, :])),
        evleaf_leaf = mlcanopy_inst.evleaf_leaf.at[p, _sl_lf, :].set(
                          jnp.array(_evleaf_new[_sl_lf, :])),
        trleaf_leaf = mlcanopy_inst.trleaf_leaf.at[p, _sl_lf, :].set(
                          jnp.array(_trleaf_new[_sl_lf, :])),
    )

    # ------------------------------------------------------------------
    # Soil fluxes — Fortran lines 358-360
    # ------------------------------------------------------------------
    mlcanopy_inst = SoilFluxes(p, mlcanopy_inst)

    # Re-read updated profiles (shair/etair/stair updated below)
    shair = mlcanopy_inst.shair_profile
    etair = mlcanopy_inst.etair_profile
    stair = mlcanopy_inst.stair_profile
    # tair/eair were not modified by LeafFluxes or SoilFluxes; reuse numpy
    _eair_final = _eair_lf_p
    _tair_final = _tair_solved

    # ------------------------------------------------------------------
    # Vertical sensible heat and water vapor fluxes between layers
    # Fortran lines 363-370
    # Vectorised: build "next-layer" arrays so ic=n uses thref/eref boundary.
    # ------------------------------------------------------------------
    ics = np.arange(1, n + 1)
    _tair_next = np.empty(n + 2)
    _tair_next[1:n + 1] = _tair_final[2:n + 2]   # tair[ic+1] for ic=1..n-1
    _tair_next[n] = _thref_p                        # top boundary for ic=n
    _eair_next = np.empty(n + 2)
    _eair_next[1:n + 1] = _eair_final[2:n + 2]   # eair[ic+1] for ic=1..n-1
    _eair_next[n] = _eref_p                         # top boundary for ic=n

    _shair_new = np.zeros(nlevmlcan + 1)
    _etair_new = np.zeros(nlevmlcan + 1)
    _shair_new[ics] = -_cpair_p * (_tair_next[ics] - _tair_final[ics]) * _gac_p[ics]
    _etair_new[ics] = -(_eair_next[ics] - _eair_final[ics]) / _pref_p * _gac_p[ics]

    # ------------------------------------------------------------------
    # Canopy air storage flux — Fortran lines 373-379
    # ------------------------------------------------------------------
    storage_sh = np.zeros(nlevmlcan + 1)
    storage_et = np.zeros(nlevmlcan + 1)
    storage_sh[ics] = _cpair_p * (_tair_final[ics] - _tair_bef_p[ics]) * rho_dz_over_dt[ics]
    storage_et[ics] = (_eair_final[ics] - _eair_bef_p[ics]) / _pref_p * rho_dz_over_dt[ics]
    _stair_new = np.zeros(nlevmlcan + 1)
    _stair_new[ics] = storage_sh[ics] + storage_et[ics] * lambda_

    # Batch write-back: 3 bulk JAX ops instead of 3n per-element writes
    _sl = slice(1, n + 1)
    mlcanopy_inst = mlcanopy_inst._replace(
        shair_profile = shair.at[p, _sl].set(jnp.array(_shair_new[_sl])),
        etair_profile = etair.at[p, _sl].set(jnp.array(_etair_new[_sl])),
        stair_profile = stair.at[p, _sl].set(jnp.array(_stair_new[_sl])),
    )

    # ------------------------------------------------------------------
    # Source fluxes from implicit solution — Fortran lines 387-398
    # ------------------------------------------------------------------
    shsrc = np.zeros(nlevmlcan + 1)
    etsrc = np.zeros(nlevmlcan + 1)
    stveg = np.zeros(nlevmlcan + 1)

    for ic in range(1, n + 1):
        if _dpai_p[ic] > 0.0:
            for il in [isun, isha]:
                shsrc[ic] += _cpair_p * (tleaf_implic[ic, il] - _tair_final[ic]) * gleaf_sh[ic, il]
                esat_l, desat_l = SatVap(_tleaf_bef_p[ic, il])
                etsrc[ic] += (
                    (esat_l + desat_l * (tleaf_implic[ic, il] - _tleaf_bef_p[ic, il])
                     - _eair_final[ic]) / _pref_p * gleaf_et[ic, il]
                )
                stveg[ic] += heatcap[ic, il] * (tleaf_implic[ic, il] - _tleaf_bef_p[ic, il]) / dtime

    # Soil fluxes from implicit solution — Fortran lines 401-403
    sh0 = -_cpair_p * (_tair_final[1] - t0) * _gac0_p
    et0 = -(_eair_final[1] - e0) / _pref_p * gs0
    g0  = -_soil_tk_p / _soil_dz_p * _soil_t_p + _soil_tk_p / _soil_dz_p * t0

    # ------------------------------------------------------------------
    # Error checks — Fortran lines 407-412
    # ------------------------------------------------------------------
    mlcanopy_inst = ErrorCheck01(
        p, lambda_, shsrc, etsrc, stveg, tleaf_implic,
        sh0, et0, g0, t0, e0, mlcanopy_inst,
    )
    mlcanopy_inst = ErrorCheck02(
        p, lambda_, avail_energy, shsrc, etsrc, stveg,
        storage_sh, storage_et, sh0, et0, g0, mlcanopy_inst,
    )

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: implicit solution error checks
# ---------------------------------------------------------------------------

def ErrorCheck01(
    p: int,
    lambda_: float,
    shsrc: Array,
    etsrc: Array,
    stveg: Array,
    tleaf_implic: Array,
    sh0: float,
    et0: float,
    g0: float,
    t0: float,
    e0: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Compare the implicit solution with ``LeafFluxes`` and ``SoilFluxes``.

    Mirrors Fortran subroutine ``ErrorCheck01`` (lines 315-390).

    Args:
        p: Patch index.
        lambda_: Latent heat of vaporization (J/mol).
        shsrc: Implicit-solution canopy-layer leaf SH flux (W/m2),
            shape ``(nlevmlcan,)``.
        etsrc: Implicit-solution canopy-layer leaf ET flux
            (mol H2O/m2/s), shape ``(nlevmlcan,)``.
        stveg: Implicit-solution canopy-layer leaf storage flux (W/m2),
            shape ``(nlevmlcan,)``.
        tleaf_implic: Implicit-solution leaf temperature (K),
            shape ``(nlevmlcan, nleaf)``.
        sh0: Implicit-solution ground SH flux (W/m2).
        et0: Implicit-solution ground ET flux (mol H2O/m2/s).
        g0: Implicit-solution soil heat flux (W/m2).
        t0: Implicit-solution soil surface temperature (K).
        e0: Implicit-solution soil surface vapor pressure (Pa).
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Unchanged ``mlcanopy_inst`` (or aborts on error).
    """
    n = int(mlcanopy_inst.ncan_canopy[p])

    # Pre-extract all per-layer arrays as numpy — 7 syncs total vs ~120 in the loop
    _dpai    = np.asarray(mlcanopy_inst.dpai_profile[p])
    _fracsun = np.asarray(mlcanopy_inst.fracsun_profile[p])
    _shleaf  = np.asarray(mlcanopy_inst.shleaf_leaf[p])   # (nlevmlcan+1, nleaf+1)
    _trleaf  = np.asarray(mlcanopy_inst.trleaf_leaf[p])
    _evleaf  = np.asarray(mlcanopy_inst.evleaf_leaf[p])
    _stleaf  = np.asarray(mlcanopy_inst.stleaf_leaf[p])
    _tleaf   = np.asarray(mlcanopy_inst.tleaf_leaf[p])

    # Compare leaf fluxes — Fortran lines 358-383
    for ic in range(1, n + 1):
        if _dpai[ic] > 0.0:

            fs_ic   = float(_fracsun[ic])
            dpai_ic = float(_dpai[ic])

            # Layer fluxes from LeafFluxes — Fortran lines 361-366
            shsrc_leaf = (
                float(_shleaf[ic, isun]) * fs_ic
                + float(_shleaf[ic, isha]) * (1.0 - fs_ic)
            ) * dpai_ic

            etsrc_leaf = (
                (float(_trleaf[ic, isun]) + float(_evleaf[ic, isun])) * fs_ic
                + (float(_trleaf[ic, isha]) + float(_evleaf[ic, isha])) * (1.0 - fs_ic)
            ) * dpai_ic

            stveg_leaf = (
                float(_stleaf[ic, isun]) * fs_ic
                + float(_stleaf[ic, isha]) * (1.0 - fs_ic)
            ) * dpai_ic

            # Error checks — Fortran lines 369-383
            if abs(shsrc[ic] - shsrc_leaf) > 0.001:
                endrun(msg=' ERROR: ImplicitFluxProfileSolution: Leaf sensible heat flux error')

            if abs(lambda_ * (etsrc[ic] - etsrc_leaf)) > 0.001:
                endrun(msg=' ERROR: ImplicitFluxProfileSolution: Leaf latent heat flux error')

            if abs(stveg[ic] - stveg_leaf) > 0.001:
                endrun(msg=' ERROR: ImplicitFluxProfileSolution: Leaf heat storage error')

            if abs(float(_tleaf[ic, isun]) - tleaf_implic[ic, isun]) > 1.0e-6:
                endrun(msg=' ERROR: ImplicitFluxProfileSolution: Leaf temperature error (sunlit)')

            if abs(float(_tleaf[ic, isha]) - tleaf_implic[ic, isha]) > 1.0e-6:
                endrun(msg=' ERROR: ImplicitFluxProfileSolution: Leaf temperature error (shaded)')

    # Compare soil fluxes — Fortran lines 386-398
    if abs(float(mlcanopy_inst.shsoi_soil[p]) - sh0) > 0.001:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Soil sensible heat flux error')

    if abs(lambda_ * (float(mlcanopy_inst.etsoi_soil[p]) - et0)) > 0.001:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Soil latent heat flux error')

    if abs(float(mlcanopy_inst.gsoi_soil[p]) - g0) > 0.001:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Soil heat flux error')

    if abs(float(mlcanopy_inst.tg_soil[p]) - t0) > 1.0e-6:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Soil surface temperature error')

    if abs(float(mlcanopy_inst.eg_soil[p]) - e0) > 1.0e-6:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Soil surface vapor pressure error')

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: conservation error checks
# ---------------------------------------------------------------------------

def ErrorCheck02(
    p: int,
    lambda_: float,
    avail_energy: Array,
    shsrc: Array,
    etsrc: Array,
    stveg: Array,
    storage_sh: Array,
    storage_et: Array,
    sh0: float,
    et0: float,
    g0: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Conservation error checks for the implicit flux-profile solution.

    Mirrors Fortran subroutine ``ErrorCheck02`` (lines 392-470).

    Args:
        p: Patch index.
        lambda_: Latent heat of vaporization (J/mol).
        avail_energy: Available energy per leaf type (W/m2),
            shape ``(nlevmlcan, nleaf)``.
        shsrc: Canopy-layer leaf SH flux (W/m2), shape ``(nlevmlcan,)``.
        etsrc: Canopy-layer leaf ET flux (mol H2O/m2/s),
            shape ``(nlevmlcan,)``.
        stveg: Canopy-layer leaf storage flux (W/m2),
            shape ``(nlevmlcan,)``.
        storage_sh: Air SH storage flux (W/m2), shape ``(nlevmlcan,)``.
        storage_et: Air ET storage flux (mol H2O/m2/s),
            shape ``(nlevmlcan,)``.
        sh0: Ground SH flux (W/m2).
        et0: Ground ET flux (mol H2O/m2/s).
        g0: Soil heat flux (W/m2).
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Unchanged ``mlcanopy_inst`` (or aborts on error).
    """
    n      = int(mlcanopy_inst.ncan_canopy[p])
    ntop_p = int(mlcanopy_inst.ntop_canopy[p])

    # Pre-extract shair/etair as numpy — 2 syncs vs 4n in the loops below
    _shair = np.asarray(mlcanopy_inst.shair_profile[p])
    _etair = np.asarray(mlcanopy_inst.etair_profile[p])

    # Vegetation flux energy balance — Fortran lines 431-436
    for ic in range(1, n + 1):
        err = (
            avail_energy[ic, isun] + avail_energy[ic, isha]
            - shsrc[ic] - lambda_ * etsrc[ic] - stveg[ic]
        )
        if abs(err) > 0.001:
            endrun(msg=' ERROR: ImplicitFluxProfileSolution: Leaf energy balance error')

    # Flux conservation at each layer — Fortran lines 439-455
    for ic in range(1, n + 1):
        if ic == 1:
            err = storage_sh[ic] - (sh0 + shsrc[ic] - float(_shair[ic]))
        else:
            err = storage_sh[ic] - (float(_shair[ic - 1]) + shsrc[ic] - float(_shair[ic]))
        if abs(err) > 0.001:
            endrun(msg=' ERROR: ImplicitFluxProfileSolution: Sensible heat layer conservation error')

        if ic == 1:
            err = storage_et[ic] - (et0 + etsrc[ic] - float(_etair[ic]))
        else:
            err = storage_et[ic] - (float(_etair[ic - 1]) + etsrc[ic] - float(_etair[ic]))
        err = err * lambda_
        if abs(err) > 0.001:
            endrun(msg=' ERROR: ImplicitFluxProfileSolution: Latent heat layer conservation error')

    # Canopy sensible heat conservation — Fortran lines 458-465
    sum_src_sh     = float(np.sum(shsrc[1:ntop_p + 1]))
    sum_storage_sh = float(np.sum(storage_sh[1:ntop_p + 1]))
    err = (sh0 + sum_src_sh - sum_storage_sh) - float(_shair[ntop_p])
    if abs(err) > 0.001:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Sensible heat canopy conservation error')

    # Canopy latent heat conservation — Fortran lines 468-475
    sum_src_et     = float(np.sum(etsrc[1:ntop_p + 1]))
    sum_storage_et = float(np.sum(storage_et[1:ntop_p + 1]))
    err = ((et0 + sum_src_et - sum_storage_et) - float(_etair[ntop_p])) * lambda_
    if abs(err) > 0.001:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Latent heat canopy conservation error')

    # Ground energy balance — Fortran lines 478-481
    err = float(mlcanopy_inst.rnsoi_soil[p]) - sh0 - lambda_ * et0 - g0
    if abs(err) > 0.001:
        endrun(msg=' ERROR: ImplicitFluxProfileSolution: Ground temperature energy balance error')

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: well-mixed canopy assumption
# ---------------------------------------------------------------------------

def WellMixed(
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Set canopy scalar profiles to reference-height values (well-mixed
    assumption) or read them from a dataset.

    Mirrors Fortran subroutine ``WellMixed`` (lines 484-580).

    Args:
        num_filter: Number of patches in the filter.
        filter: 1-D array of patch indices.
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Updated :class:`mlcanopy_type` with profiles set and leaf/soil
        fluxes computed.
    """
    # Unpack inputs
    uref      = mlcanopy_inst.uref_forcing
    tref      = mlcanopy_inst.tref_forcing
    eref      = mlcanopy_inst.eref_forcing
    co2ref    = mlcanopy_inst.co2ref_forcing
    qref      = mlcanopy_inst.qref_forcing
    ncan      = mlcanopy_inst.ncan_canopy
    wind_data = mlcanopy_inst.wind_data_profile
    tair_data = mlcanopy_inst.tair_data_profile
    eair_data = mlcanopy_inst.eair_data_profile

    uaf   = mlcanopy_inst.uaf_canopy
    taf   = mlcanopy_inst.taf_canopy
    qaf   = mlcanopy_inst.qaf_canopy
    gac0  = mlcanopy_inst.gac0_soil
    wind  = mlcanopy_inst.wind_profile
    tair  = mlcanopy_inst.tair_profile
    eair  = mlcanopy_inst.eair_profile
    cair  = mlcanopy_inst.cair_profile
    gac   = mlcanopy_inst.gac_profile
    shair = mlcanopy_inst.shair_profile
    etair = mlcanopy_inst.etair_profile
    stair = mlcanopy_inst.stair_profile
    mflx  = mlcanopy_inst.mflx_profile

    for fp in range(num_filter):
        p = int(filter[fp])
        n = int(ncan[p])

        for ic in range(n):

            # CO2 profile — Fortran line 535
            cair = cair.at[p, ic].set(co2ref[p])

            # Scalar profiles — Fortran lines 538-554
            if flux_profile_type == 0:
                # Well-mixed assumption
                wind = wind.at[p, ic].set(uref[p])
                tair = tair.at[p, ic].set(tref[p])
                eair = eair.at[p, ic].set(eref[p])
            elif flux_profile_type == -1:
                # Dataset profiles
                wind = wind.at[p, ic].set(wind_data[p, ic])
                tair = tair.at[p, ic].set(tair_data[p, ic])
                eair = eair.at[p, ic].set(eair_data[p, ic])

            # Vertical fluxes set to zero — Fortran lines 557-560
            shair = shair.at[p, ic].set(0.0)
            etair = etair.at[p, ic].set(0.0)
            stair = stair.at[p, ic].set(0.0)
            mflx  = mflx.at[p, ic].set(0.0)

            # Update state before calling leaf flux routines
            mlcanopy_inst = mlcanopy_inst._replace(
                wind_profile=wind, tair_profile=tair, eair_profile=eair,
                cair_profile=cair, shair_profile=shair, etair_profile=etair,
                stair_profile=stair, mflx_profile=mflx,
            )

            # Leaf fluxes — Fortran lines 563-564
            mlcanopy_inst = LeafFluxes(p, ic, isun, mlcanopy_inst)
            mlcanopy_inst = LeafFluxes(p, ic, isha, mlcanopy_inst)

            # Re-read any fields written by LeafFluxes
            wind  = mlcanopy_inst.wind_profile
            tair  = mlcanopy_inst.tair_profile
            eair  = mlcanopy_inst.eair_profile
            shair = mlcanopy_inst.shair_profile
            etair = mlcanopy_inst.etair_profile
            stair = mlcanopy_inst.stair_profile
            mflx  = mlcanopy_inst.mflx_profile

        # Soil conductance: large resistance -> small soil fluxes — Fortran lines 568-569
        # gac0 = (1/100) * 42.3  mol/m2/s
        gac0 = gac0.at[p].set((1.0 / 100.0) * 42.3)
        mlcanopy_inst = mlcanopy_inst._replace(gac0_soil=gac0)
        mlcanopy_inst = SoilFluxes(p, mlcanopy_inst)

        # Output-only fields — Fortran lines 573-578
        uaf = uaf.at[p].set(uref[p])
        taf = taf.at[p].set(tref[p])
        qaf = qaf.at[p].set(qref[p])

        gac0 = mlcanopy_inst.gac0_soil
        gac  = mlcanopy_inst.gac_profile

        for ic in range(n):
            gac = gac.at[p, ic].set((1.0 / 10.0) * 42.3)  # small non-zero resistance

        mlcanopy_inst = mlcanopy_inst._replace(
            uaf_canopy=uaf,
            taf_canopy=taf,
            qaf_canopy=qaf,
            gac_profile=gac,
        )

    return mlcanopy_inst