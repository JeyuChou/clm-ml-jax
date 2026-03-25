"""
JAX translation of MLSolarRadiationMod Fortran module.

Solar radiation transfer through the multilayer canopy.
Provides one public driver (:func:`SolarRadiation`) and two private
solvers (:func:`_Norman`, :func:`_TwoStream`).

Original Fortran module: MLSolarRadiationMod
Fortran lines 1-490
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import jax.numpy as jnp
from jax import Array

from clm_src_main.abortutils import endrun                           # noqa: F401
from clm_src_main.clm_varctl import iulog                           # noqa: F401
from clm_src_main.decompMod import bounds_type                      # noqa: F401
from clm_src_main.PatchType import patch                            # noqa: F401
from clm_src_main.pftconMod import pftcon                           # noqa: F401
from multilayer_canopy.MLpftconMod import MLpftcon                       # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type           # noqa: F401
from clm_src_main.clm_varcon import rpi as pi                       # noqa: F401
from clm_src_main.clm_varpar import numrad, ivis                    # noqa: F401
from multilayer_canopy.MLclm_varcon import chil_max, chil_min, kb_max, J_to_umol  # noqa: F401
from multilayer_canopy.MLclm_varctl import light_type, leaf_optics_type  # noqa: F401
from multilayer_canopy.MLclm_varpar import nlevmlcan, isun, isha          # noqa: F401
from multilayer_canopy.MLMathToolsMod import tridiag                     # noqa: F401


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def SolarRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Solar radiation transfer through the multilayer canopy.

    Mirrors Fortran subroutine ``SolarRadiation`` (lines 28-195).

    References
    ----------
    Bonan (2019) *Climate Change and Terrestrial Ecosystem Modeling*,
    Chapter 14.
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eqs. (1)-(5).

    Execution sequence
    ------------------
    1. For each patch, zero all layer optical variables.
    2. Loop over leaf layers (``ntop`` down to ``nbot``): compute
       leaf/stem reflectance (``rho``), transmittance (``tau``),
       scattering (``omega``), leaf angle distribution (``chil``,
       ``phi1``, ``phi2``, ``gdir``), direct beam extinction
       coefficient ``kb``, clumping factor, single-layer direct
       transmittance ``tb``, diffuse transmittance ``td`` (9-angle
       numerical integration), cumulative direct beam transmittance
       ``tbi``, sunlit fraction ``fracsun``, and two-stream parameters
       ``avmu``, ``betad``, ``betab``.
    3. Compute ``tbi(p,0)`` (transmittance onto the ground).
    4. Dispatch to :func:`_Norman` (``light_type == 1``) or
       :func:`_TwoStream` (``light_type == 2``).
    5. Convert PAR absorbed by leaves from W/m2 to umol/m2/s via
       ``J_to_umol``.

    Optical properties follow ``leaf_optics_type``:

    - **0** (constant with height): weighted average of leaf and stem
      properties by ``wl = dlai/dpai``, ``ws = dsai/dpai``.
    - **1** (vary with height): fatal — vertical profiles not
      implemented.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Multilayer canopy container; optical diagnostics,
            flux profiles, and absorbed radiation fields are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # ------------------------------------------------------------------
    # Unpack inputs (Fortran associate block, lines 65-95)
    # ------------------------------------------------------------------
    xl        = pftcon.xl
    rhol      = pftcon.rhol
    taul      = pftcon.taul
    rhos      = pftcon.rhos
    taus      = pftcon.taus
    clump_fac = MLpftcon.clump_fac

    solar_zen = mlcanopy_inst.solar_zen_forcing
    ncan      = mlcanopy_inst.ncan_canopy
    ntop      = mlcanopy_inst.ntop_canopy
    nbot      = mlcanopy_inst.nbot_canopy
    dlai      = mlcanopy_inst.dlai_profile
    dsai      = mlcanopy_inst.dsai_profile
    dpai      = mlcanopy_inst.dpai_profile

    fracsun = mlcanopy_inst.fracsun_profile
    kb      = mlcanopy_inst.kb_profile
    tb      = mlcanopy_inst.tb_profile
    td      = mlcanopy_inst.td_profile
    tbi     = mlcanopy_inst.tbi_profile
    apar    = mlcanopy_inst.apar_leaf

    # ------------------------------------------------------------------
    # Working numpy arrays for optical properties (avoid JAX XLA syncs)
    # ------------------------------------------------------------------
    n_idx = bounds.endp + 1
    n_lev = nlevmlcan + 1
    n_rad = numrad + 1
    _avmu    = np.zeros((n_idx, n_lev))
    _betad   = np.zeros((n_idx, n_lev, n_rad))
    _betab   = np.zeros((n_idx, n_lev, n_rad))
    _cf_ic   = np.zeros((n_idx, n_lev))
    _rho_arr = np.zeros((n_idx, n_lev, n_rad))
    _tau_arr = np.zeros((n_idx, n_lev, n_rad))
    _om_arr  = np.zeros((n_idx, n_lev, n_rad))

    # ------------------------------------------------------------------
    # Calculate canopy layer optical properties — Fortran lines 100-182
    # ------------------------------------------------------------------
    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])
        zen = float(solar_zen[p])
        cos_zen = math.cos(zen)
        _ncan = int(ncan[p])
        _ntop = int(ntop[p])
        _nbot = int(nbot[p])

        # Pre-extract read-only profile inputs as numpy (one sync each)
        _dpai_p = np.asarray(dpai[p])
        _dlai_p = np.asarray(dlai[p])
        _dsai_p = np.asarray(dsai[p])

        # Per-patch output arrays (numpy, no XLA syncs during computation)
        _kb_p     = np.zeros(_ncan + 2)
        _fracsun_p = np.zeros(_ncan + 2)
        _tb_p     = np.zeros(_ncan + 2)
        _td_p     = np.zeros(_ncan + 2)
        _tbi_p    = np.zeros(_ncan + 2)   # index 0 = ground, 1.._ncan = canopy

        # Extract scalar PFT values once
        if leaf_optics_type == 0:
            _xl_pft = float(xl[pft])
            _cf_pft = float(clump_fac[pft])
            _rhol_ib = [float(rhol[pft, ib]) for ib in range(n_rad)]
            _taul_ib = [float(taul[pft, ib]) for ib in range(n_rad)]
            _rhos_ib = [float(rhos[pft, ib]) for ib in range(n_rad)]
            _taus_ib = [float(taus[pft, ib]) for ib in range(n_rad)]
        else:
            endrun(msg=' ERROR: SolarRadiation: need to specify vertical profile for rho & tau')
            _xl_pft = 0.01; _cf_pft = 1.0
            _rhol_ib = _taul_ib = _rhos_ib = _taus_ib = [1.0e-6] * n_rad

        # Layer-by-layer optical properties — Fortran: do ic = ntop, nbot, -1
        for ic in range(_ntop, _nbot - 1, -1):

            dpai_ic = _dpai_p[ic]
            dlai_ic = _dlai_p[ic]
            dsai_ic = _dsai_p[ic]
            wl = dlai_ic / dpai_ic
            ws = dsai_ic / dpai_ic

            # Reflectance, transmittance, scattering — Fortran lines 121-132
            for ib in range(1, numrad + 1):
                r = max(_rhol_ib[ib] * wl + _rhos_ib[ib] * ws, 1.0e-6)
                t = max(_taul_ib[ib] * wl + _taus_ib[ib] * ws, 1.0e-6)
                _rho_arr[p, ic, ib] = r
                _tau_arr[p, ic, ib] = t
                _om_arr[p, ic, ib]  = r + t

            # Leaf angle distribution — Fortran lines 134-141
            chil_ic = _xl_pft
            chil_ic = min(max(chil_ic, chil_min), chil_max)
            if abs(chil_ic) <= 0.01:
                chil_ic = 0.01

            # Ross-Goudriaan phi1, phi2, gdir — Fortran lines 143-148
            p1 = 0.5 - 0.633 * chil_ic - 0.330 * chil_ic * chil_ic
            p2 = 0.877 * (1.0 - 2.0 * p1)
            gd = p1 + p2 * cos_zen

            # Direct beam extinction coefficient — Fortran lines 150-152
            kb_ic = min(gd / cos_zen, kb_max)
            _kb_p[ic] = kb_ic

            # Clumping factor — Fortran lines 154-159
            cf = _cf_pft
            _cf_ic[p, ic] = cf

            # Direct beam single-layer transmittance — Fortran line 161
            _tb_p[ic] = math.exp(-kb_ic * dpai_ic * cf)

            # Diffuse transmittance (9-angle integration) — Fortran lines 163-170
            td_ic = 0.0
            for j in range(1, 10):
                angle = (5.0 + (j - 1) * 10.0) * pi / 180.0
                gdirj = p1 + p2 * math.cos(angle)
                td_ic += (
                    math.exp(-gdirj / math.cos(angle) * dpai_ic * cf)
                    * math.sin(angle) * math.cos(angle)
                )
            _td_p[ic] = td_ic * 2.0 * (10.0 * pi / 180.0)

            # Cumulative direct beam transmittance tbi — Fortran lines 172-177
            if ic == _ntop:
                _tbi_p[ic] = 1.0
            else:
                # Read from numpy — no XLA sync
                _tbi_p[ic] = _tbi_p[ic + 1] * math.exp(
                    -_kb_p[ic + 1] * _dpai_p[ic + 1] * _cf_ic[p, ic + 1]
                )

            # Sunlit fraction — Fortran lines 179-188
            tbi_ic = _tbi_p[ic]
            fracsun_ic = (
                tbi_ic / (kb_ic * dpai_ic)
                * (1.0 - math.exp(-kb_ic * cf * dpai_ic))
            )
            if fracsun_ic <= 0.0:
                endrun(msg=' ERROR: SolarRadiation: fracsun is too small')
            if (1.0 - fracsun_ic) <= 0.0:
                endrun(msg=' ERROR: SolarRadiation: fracsha is too small')
            _fracsun_p[ic] = fracsun_ic

            # Two-stream avmu — Fortran lines 190-194
            avmu_ic = (1.0 - p1 / p2 * math.log((p1 + p2) / p1)) / p2
            _avmu[p, ic] = avmu_ic

            # betad, betab — Fortran lines 196-209
            for ib in range(1, numrad + 1):
                om   = _om_arr[p, ic, ib]
                r_ic = _rho_arr[p, ic, ib]
                t_ic = _tau_arr[p, ic, ib]
                _betad[p, ic, ib] = (
                    0.5 / om * (r_ic + t_ic + (r_ic - t_ic) * ((1.0 + chil_ic) / 2.0) ** 2)
                )
                tmp0 = gd + p2 * cos_zen
                tmp1 = p1 * cos_zen
                tmp2 = 1.0 - tmp1 / tmp0 * math.log((tmp1 + tmp0) / tmp1)
                asu  = 0.5 * om * gd / tmp0 * tmp2
                _betab[p, ic, ib] = (1.0 + avmu_ic * kb_ic) / (om * avmu_ic * kb_ic) * asu

        # tbi onto ground — Fortran lines 211-213
        _tbi_p[0] = _tbi_p[_nbot] * math.exp(
            -_kb_p[_nbot] * _dpai_p[_nbot] * _cf_ic[p, _nbot]
        )

        # Bulk JAX write-back — one write per field per patch
        _sl  = slice(1, _ncan + 1)
        _sl0 = slice(0, _ncan + 1)
        kb      = kb.at[p,      _sl].set(jnp.array(_kb_p[1:_ncan + 1]))
        fracsun = fracsun.at[p, _sl].set(jnp.array(_fracsun_p[1:_ncan + 1]))
        tb      = tb.at[p,      _sl].set(jnp.array(_tb_p[1:_ncan + 1]))
        td      = td.at[p,      _sl].set(jnp.array(_td_p[1:_ncan + 1]))
        tbi     = tbi.at[p,     _sl0].set(jnp.array(_tbi_p[0:_ncan + 1]))

    # ------------------------------------------------------------------
    # Commit optical properties before radiative transfer
    # ------------------------------------------------------------------
    mlcanopy_inst = mlcanopy_inst._replace(
        fracsun_profile = fracsun,
        kb_profile      = kb,
        tb_profile      = tb,
        td_profile      = td,
        tbi_profile     = tbi,
    )

    # ------------------------------------------------------------------
    # Radiative transfer solver — Fortran lines 216-222
    # ------------------------------------------------------------------
    if light_type == 1:
        mlcanopy_inst = _Norman(
            bounds, num_filter, filter_patch,
            _rho_arr, _tau_arr, _om_arr,
            mlcanopy_inst,
        )
    elif light_type == 2:
        mlcanopy_inst = _TwoStream(
            bounds, num_filter, filter_patch,
            _om_arr, _avmu, _betad, _betab, _cf_ic,
            mlcanopy_inst,
        )
    else:
        endrun(msg=' ERROR: SolarRadiation: light_type not valid')

    # ------------------------------------------------------------------
    # APAR: W/m2 → umol/m2/s — Fortran lines 224-229 (vectorized)
    # ------------------------------------------------------------------
    swleaf = mlcanopy_inst.swleaf_leaf
    apar   = mlcanopy_inst.apar_leaf
    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1])
        _ncan = int(ncan[p])
        _sl = slice(1, _ncan + 1)
        # Bulk write: apar[p, 1:ncan+1, :] = swleaf[p, 1:ncan+1, :, ivis] * J_to_umol
        apar = apar.at[p, _sl, :].set(swleaf[p, _sl, :, ivis] * J_to_umol)

    mlcanopy_inst = mlcanopy_inst._replace(apar_leaf = apar)
    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: Norman (1979) radiative transfer
# ---------------------------------------------------------------------------

def _Norman(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    rho: Array,
    tau: Array,
    omega: Array,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Norman (1979) tridiagonal radiative transfer through the canopy.

    Mirrors Fortran subroutine ``Norman`` (private, lines 232-370).

    For each waveband ``ib`` and each patch ``p``, sets up a tridiagonal
    system of ``2*(nbot-to-ntop layers + soil)`` equations for the
    upward and downward diffuse fluxes at each layer interface and
    solves it via :func:`MLMathToolsMod.tridiag`. The ordering of
    equations follows the Fortran exactly: soil upward (m=1), soil
    downward (m=2), then for ``ic = nbot, ..., ntop-1`` upward (odd)
    and downward (even), finally top-layer upward and downward.

    After solving, absorbed direct-beam and diffuse fluxes per layer
    are partitioned into sunlit (``swleaf[..,isun,..]``) and shaded
    (``swleaf[..,isha,..]``) components per unit leaf area. Energy
    conservation is checked to within 1e-3 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch filter.
        rho: Leaf/stem reflectance ``(p, ic, ib)``.
        tau: Leaf/stem transmittance ``(p, ic, ib)``.
        omega: Scattering coefficient ``(p, ic, ib)``.
        mlcanopy_inst: Canopy container; radiation flux fields updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    neq = (nlevmlcan + 1) * 2    # Fortran: parameter neq = (nlevmlcan+1)*2

    swskyb  = mlcanopy_inst.swskyb_forcing
    swskyd  = mlcanopy_inst.swskyd_forcing
    ncan    = mlcanopy_inst.ncan_canopy
    ntop    = mlcanopy_inst.ntop_canopy
    nbot    = mlcanopy_inst.nbot_canopy
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    dpai    = mlcanopy_inst.dpai_profile
    fracsun = mlcanopy_inst.fracsun_profile
    tb      = mlcanopy_inst.tb_profile
    td      = mlcanopy_inst.td_profile
    tbi     = mlcanopy_inst.tbi_profile

    swveg    = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    albcan   = mlcanopy_inst.albcan_canopy
    swsoi    = mlcanopy_inst.swsoi_soil
    swleaf   = mlcanopy_inst.swleaf_leaf
    swupw    = mlcanopy_inst.swupw_profile
    swdwn    = mlcanopy_inst.swdwn_profile
    swbeam   = mlcanopy_inst.swbeam_profile

    # Combined tridiagonal + flux loop — avoids write-then-read-back XLA syncs
    # by keeping swupw/swdwn as numpy until the final bulk write-back.
    for ib in range(1, numrad + 1):                    # Fortran: do ib = 1, numrad
        for fp in range(1, num_filter + 1):
            p = int(filter_patch[fp - 1])
            _ncan = int(ncan[p])
            _ntop = int(ntop[p])
            _nbot = int(nbot[p])

            # Pre-extract read-only JAX arrays as numpy — 5 syncs total
            _td_p      = np.asarray(td[p])        # shape (nlevmlcan+1,)
            _tb_p      = np.asarray(tb[p])
            _tbi_p     = np.asarray(tbi[p])
            _dpai_p    = np.asarray(dpai[p])
            _fracsun_p = np.asarray(fracsun[p])

            # Scalar inputs — 4 syncs
            _swskyb_ib  = float(swskyb[p, ib])
            _swskyd_ib  = float(swskyd[p, ib])
            _albsoib_ib = float(albsoib[p, ib])
            _albsoid_ib = float(albsoid[p, ib])

            # Zero swleaf for all layers (bulk — 1 JAX op instead of 2*_ncan)
            swleaf = swleaf.at[p, 1:_ncan + 1, :, ib].set(0.0)

            # ----------------------------------------------------------------
            # Build tridiagonal system — Fortran lines 284-340
            # (all reads from numpy now; no JAX syncs in this section)
            # ----------------------------------------------------------------
            atri = [0.0] * (neq + 1)
            btri = [0.0] * (neq + 1)
            ctri = [0.0] * (neq + 1)
            dtri = [0.0] * (neq + 1)

            m = 0

            # Soil: upward flux — Fortran lines 287-290
            m += 1
            atri[m] = 0.0
            btri[m] = 1.0
            ctri[m] = -_albsoid_ib
            dtri[m] = _swskyb_ib * float(_tbi_p[0]) * _albsoib_ib

            # Soil: downward flux — Fortran lines 292-301
            td_nb  = float(_td_p[_nbot])
            rho_nb = float(rho[p, _nbot, ib])   # numpy — no JAX sync
            tau_nb = float(tau[p, _nbot, ib])   # numpy — no JAX sync
            tb_nb  = float(_tb_p[_nbot])
            tbi_nb = float(_tbi_p[_nbot])
            refld = (1.0 - td_nb) * rho_nb
            trand = (1.0 - td_nb) * tau_nb + td_nb
            aic   = refld - trand * trand / refld
            bic   = trand / refld
            m += 1
            atri[m] = -aic
            btri[m] = 1.0
            ctri[m] = -bic
            dtri[m] = _swskyb_ib * tbi_nb * (1.0 - tb_nb) * (tau_nb - rho_nb * bic)

            # Leaf layers except top — Fortran lines 303-326
            for ic in range(_nbot, _ntop):             # Fortran: do ic = nbot, ntop-1

                # Upward flux
                td_ic  = float(_td_p[ic])
                rho_ic = float(rho[p, ic, ib])
                tau_ic = float(tau[p, ic, ib])
                refld = (1.0 - td_ic) * rho_ic
                trand = (1.0 - td_ic) * tau_ic + td_ic
                fic   = refld - trand * trand / refld
                eic   = trand / refld
                m += 1
                atri[m] = -eic
                btri[m] = 1.0
                ctri[m] = -fic
                dtri[m] = (_swskyb_ib * float(_tbi_p[ic])
                            * (1.0 - float(_tb_p[ic])) * (rho_ic - tau_ic * eic))

                # Downward flux
                ic1     = ic + 1
                td_ic1  = float(_td_p[ic1])
                rho_ic1 = float(rho[p, ic1, ib])
                tau_ic1 = float(tau[p, ic1, ib])
                refld = (1.0 - td_ic1) * rho_ic1
                trand = (1.0 - td_ic1) * tau_ic1 + td_ic1
                aic   = refld - trand * trand / refld
                bic   = trand / refld
                m += 1
                atri[m] = -aic
                btri[m] = 1.0
                ctri[m] = -bic
                dtri[m] = (_swskyb_ib * float(_tbi_p[ic1])
                            * (1.0 - float(_tb_p[ic1])) * (tau_ic1 - rho_ic1 * bic))

            # Top layer: upward flux — Fortran lines 328-337
            ic = _ntop
            td_ic  = float(_td_p[ic])
            rho_ic = float(rho[p, ic, ib])
            tau_ic = float(tau[p, ic, ib])
            refld = (1.0 - td_ic) * rho_ic
            trand = (1.0 - td_ic) * tau_ic + td_ic
            fic   = refld - trand * trand / refld
            eic   = trand / refld
            m += 1
            atri[m] = -eic
            btri[m] = 1.0
            ctri[m] = -fic
            dtri[m] = (_swskyb_ib * float(_tbi_p[ic])
                        * (1.0 - float(_tb_p[ic])) * (rho_ic - tau_ic * eic))

            # Top layer: downward flux — Fortran lines 339-343
            m += 1
            atri[m] = 0.0
            btri[m] = 1.0
            ctri[m] = 0.0
            dtri[m] = _swskyd_ib

            # Solve — Fortran line 345
            utri = tridiag(atri, btri, ctri, dtri, m)

            # ----------------------------------------------------------------
            # Unpack solution into numpy arrays — no JAX syncs yet
            # ----------------------------------------------------------------
            _swupw_vals = np.zeros(_ncan + 2)   # indices 0.._ncan
            _swdwn_vals = np.zeros(_ncan + 2)
            m_sol = 0
            m_sol += 1;  _swupw_vals[0] = utri[m_sol]
            m_sol += 1;  _swdwn_vals[0] = utri[m_sol]
            for ic in range(_nbot, _ntop + 1):
                m_sol += 1;  _swupw_vals[ic] = utri[m_sol]
                m_sol += 1;  _swdwn_vals[ic] = utri[m_sol]

            # ----------------------------------------------------------------
            # Compute fluxes in numpy/Python — Fortran lines 365-430
            # (no JAX reads/writes until bulk write-back below)
            # ----------------------------------------------------------------

            # Ground absorption — Fortran lines 368-372
            _swbeam_0  = float(_tbi_p[0]) * _swskyb_ib
            _swsoi_ib  = _swbeam_0 * (1.0 - _albsoib_ib) + _swdwn_vals[0] * (1.0 - _albsoid_ib)

            # Per-layer accumulators
            _swbeam_vals    = np.zeros(_ncan + 2)
            _swleaf_sun_v   = np.zeros(_ncan + 2)
            _swleaf_sha_v   = np.zeros(_ncan + 2)
            _swbeam_vals[0] = _swbeam_0
            _swveg_acc    = 0.0
            _swvegsun_acc = 0.0
            _swvegsha_acc = 0.0

            for ic in range(_nbot, _ntop + 1):    # Fortran: do ic = nbot, ntop
                _swbeam_ic = float(_tbi_p[ic]) * _swskyb_ib
                _om_ic     = float(omega[p, ic, ib])   # numpy — no JAX sync
                _swabsb_ic = _swbeam_ic * (1.0 - float(_tb_p[ic])) * (1.0 - _om_ic)

                icm1 = 0 if ic == _nbot else ic - 1
                _swabsd_ic = ((_swdwn_vals[ic] + _swupw_vals[icm1])
                               * (1.0 - float(_td_p[ic])) * (1.0 - _om_ic))

                _fs   = float(_fracsun_p[ic])
                _swsha = _swabsd_ic * (1.0 - _fs)
                _swsun = _swabsd_ic * _fs + _swabsb_ic

                _dpai_ic = float(_dpai_p[ic])
                _swleaf_sun_v[ic] = _swsun / (_fs * _dpai_ic)
                _swleaf_sha_v[ic] = _swsha / ((1.0 - _fs) * _dpai_ic)
                _swbeam_vals[ic]  = _swbeam_ic
                _swveg_acc    += _swabsb_ic + _swabsd_ic
                _swvegsun_acc += _swsun
                _swvegsha_acc += _swsha

            # Albedo — Fortran lines 410-414
            _suminc    = _swskyb_ib + _swskyd_ib
            _albcan_ib = float(_swupw_vals[_ntop]) / _suminc if _suminc > 0.0 else 0.0

            # Conservation checks — Fortran lines 416-426 (pure Python, no JAX)
            _sumref = _albcan_ib * _suminc
            _sumabs = _suminc - _sumref
            _err = _sumabs - (_swveg_acc + _swsoi_ib)
            if abs(_err) > 1.0e-3:
                endrun(msg='ERROR: Norman: total solar conservation error')
            _err2 = (_swvegsun_acc + _swvegsha_acc) - _swveg_acc
            if abs(_err2) > 1.0e-3:
                endrun(msg='ERROR: Norman: sunlit/shade solar conservation error')

            # DEBUG: print SW diagnostics for first few calls
            # ----------------------------------------------------------------
            # Bulk JAX write-back — ~10 ops per (ib, p) instead of ~750
            # ----------------------------------------------------------------
            _sl         = slice(0, _ncan + 1)
            _sl_layers  = slice(_nbot, _ntop + 1)
            swupw  = swupw.at[p, _sl, ib].set(jnp.array(_swupw_vals[:_ncan + 1]))
            swdwn  = swdwn.at[p, _sl, ib].set(jnp.array(_swdwn_vals[:_ncan + 1]))
            swbeam = swbeam.at[p, _sl, ib].set(jnp.array(_swbeam_vals[:_ncan + 1]))
            swleaf = swleaf.at[p, _sl_layers, isun, ib].set(
                jnp.array(_swleaf_sun_v[_nbot:_ntop + 1]))
            swleaf = swleaf.at[p, _sl_layers, isha, ib].set(
                jnp.array(_swleaf_sha_v[_nbot:_ntop + 1]))
            swsoi    = swsoi.at[p, ib].set(_swsoi_ib)
            swveg    = swveg.at[p, ib].set(_swveg_acc)
            swvegsun = swvegsun.at[p, ib].set(_swvegsun_acc)
            swvegsha = swvegsha.at[p, ib].set(_swvegsha_acc)
            albcan   = albcan.at[p, ib].set(_albcan_ib)

        # end patch loop
    # end waveband loop

    return mlcanopy_inst._replace(
        swveg_canopy    = swveg,
        swvegsun_canopy = swvegsun,
        swvegsha_canopy = swvegsha,
        albcan_canopy   = albcan,
        swsoi_soil      = swsoi,
        swleaf_leaf     = swleaf,
        swupw_profile   = swupw,
        swdwn_profile   = swdwn,
        swbeam_profile  = swbeam,
    )


# ---------------------------------------------------------------------------
# Private: two-stream radiative transfer
# ---------------------------------------------------------------------------

def _TwoStream(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    omega: Array,
    avmu: Array,
    betad: Array,
    betab: Array,
    clump_fac_ic: Array,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Two-stream radiative transfer integrated over each canopy layer.

    Mirrors Fortran subroutine ``TwoStream`` (private, lines 372-490).

    References
    ----------
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eqs. (1)-(5).

    The algorithm proceeds in two sweeps:

    **Bottom-to-top sweep**: For each layer compute analytic two-stream
    solutions for unit direct beam (``unitb = 1``) and unit diffuse
    (``unitd = 1``) forcing. Boundary condition at the bottom is the
    soil albedo; at each subsequent layer the upward flux from the
    layer below is used. Store per-layer scattered fluxes
    ``iupwb0``, ``iupwb``, ``idwnb`` (direct) and ``iupwd0``,
    ``iupwd``, ``idwnd`` (diffuse), plus absorbed terms partitioned
    into sunlit/shaded components.

    **Top-to-bottom sweep**: Propagate the actual incident fluxes
    downward to compute ``swbeam``, ``swdwn``, ``swupw``, and per-leaf
    absorptions ``swleaf``. Accumulate canopy totals and check energy
    conservation to within 1e-6 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch filter.
        omega: Scattering coefficient ``(p, ic, ib)``.
        avmu: Average inverse diffuse optical depth ``(p, ic)``.
        betad: Diffuse upscatter parameter ``(p, ic, ib)``.
        betab: Direct beam upscatter parameter ``(p, ic, ib)``.
        clump_fac_ic: Foliage clumping index ``(p, ic)``.
        mlcanopy_inst: Canopy container; radiation fields updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    unitb: float = 1.0    # Fortran: parameter unitb = 1.0
    unitd: float = 1.0    # Fortran: parameter unitd = 1.0

    swskyb  = mlcanopy_inst.swskyb_forcing
    swskyd  = mlcanopy_inst.swskyd_forcing
    ncan    = mlcanopy_inst.ncan_canopy
    ntop    = mlcanopy_inst.ntop_canopy
    nbot    = mlcanopy_inst.nbot_canopy
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    dpai    = mlcanopy_inst.dpai_profile
    fracsun = mlcanopy_inst.fracsun_profile
    kb      = mlcanopy_inst.kb_profile
    tbi     = mlcanopy_inst.tbi_profile

    swveg    = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    albcan   = mlcanopy_inst.albcan_canopy
    swsoi    = mlcanopy_inst.swsoi_soil
    swleaf   = mlcanopy_inst.swleaf_leaf
    swupw    = mlcanopy_inst.swupw_profile
    swdwn    = mlcanopy_inst.swdwn_profile
    swbeam   = mlcanopy_inst.swbeam_profile

    n_lev = nlevmlcan + 1
    n_rad = numrad + 1
    nleaf_p1 = 3   # nleaf + 1 = 3

    # Process per patch — numpy for all intermediates, bulk JAX write-back at end
    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1])
        _ncan = int(ncan[p])
        _ntop = int(ntop[p])
        _nbot = int(nbot[p])

        # Pre-extract per-patch numpy arrays (one sync each)
        _dpai    = np.asarray(mlcanopy_inst.dpai_profile[p])
        _fracsun = np.asarray(mlcanopy_inst.fracsun_profile[p])
        _kb      = np.asarray(mlcanopy_inst.kb_profile[p])
        _tbi     = np.asarray(mlcanopy_inst.tbi_profile[p])
        _alb_b   = np.asarray(mlcanopy_inst.albsoib_soil[p])   # shape (n_rad,)
        _alb_d   = np.asarray(mlcanopy_inst.albsoid_soil[p])
        _swskyb  = np.asarray(mlcanopy_inst.swskyb_forcing[p])
        _swskyd  = np.asarray(mlcanopy_inst.swskyd_forcing[p])

        # Per-patch output numpy arrays (initialized to zero)
        _swleaf_p   = np.zeros((n_lev, nleaf_p1, n_rad))
        _swveg_p    = np.zeros(n_rad)
        _swvegsun_p = np.zeros(n_rad)
        _swvegsha_p = np.zeros(n_rad)
        _albcan_p   = np.zeros(n_rad)
        _swsoi_p    = np.zeros(n_rad)
        _swupw_p    = np.zeros((n_lev, n_rad))
        _swdwn_p    = np.zeros((n_lev, n_rad))
        _swbeam_p   = np.zeros((n_lev, n_rad))

        for ib in range(1, numrad + 1):
            # Per-band numpy slices (all numpy — no XLA syncs)
            _om  = omega[p, :, ib]
            _av  = avmu[p, :]
            _bd  = betad[p, :, ib]
            _bb  = betab[p, :, ib]
            _cf  = clump_fac_ic[p, :]

            albb = _alb_b[ib]
            albd = _alb_d[ib]

            # Per-layer work arrays
            n = _ncan + 2
            _iupwb0    = np.zeros(n)
            _iupwb     = np.zeros(n)
            _idwnb     = np.zeros(n)
            _iabsb_sun = np.zeros(n)
            _iabsb_sha = np.zeros(n)
            _iupwd0    = np.zeros(n)
            _iupwd     = np.zeros(n)
            _idwnd     = np.zeros(n)
            _iabsd_sun = np.zeros(n)
            _iabsd_sha = np.zeros(n)

            # ----------------------------------------------------------
            # Bottom-to-top sweep — Fortran lines 444-540 (pure numpy)
            # ----------------------------------------------------------
            for ic in range(_nbot, _ntop + 1):
                om_ic  = _om[ic]
                av_ic  = _av[ic]
                kb_ic  = _kb[ic]
                cf_ic  = _cf[ic]
                dp_ic  = _dpai[ic]
                bd_ic  = _bd[ic]
                bb_ic  = _bb[ic]
                tbi_ic = _tbi[ic]

                b  = (1.0 - (1.0 - bd_ic) * om_ic) / av_ic
                c  = bd_ic * om_ic / av_ic
                h  = math.sqrt(b * b - c * c)
                u  = (h - b - c) / (2.0 * h)
                v  = (h + b + c) / (2.0 * h)
                d  = om_ic * kb_ic / (h * h - kb_ic * kb_ic)   # unitb=1
                g1 = (bb_ic * kb_ic - b * bb_ic - c * (1.0 - bb_ic)) * d
                g2 = ((1.0 - bb_ic) * kb_ic + c * bb_ic + b * (1.0 - bb_ic)) * d
                s1 = math.exp(-h  * cf_ic * dp_ic)
                s2 = math.exp(-kb_ic * cf_ic * dp_ic)

                # Direct beam terms (unitb = 1)
                num1 = v * (g1 + g2 * albd + albb) * s2
                num2 = g2 * (u + v * albd) * s1
                den1 = v * (v + u * albd) / s1
                den2 = u * (u + v * albd) * s1
                n2b  = (num1 - num2) / (den1 - den2)
                n1b  = (g2 - n2b * u) / v

                a1b = (-g1       * (1.0 - s2 * s2) / (2.0 * kb_ic)
                       + n1b * u * (1.0 - s2 * s1) / (kb_ic + h)
                       + n2b * v * (1.0 - s2 / s1) / (kb_ic - h)) * tbi_ic
                a2b = ( g2       * (1.0 - s2 * s2) / (2.0 * kb_ic)
                       - n1b * v * (1.0 - s2 * s1) / (kb_ic + h)
                       - n2b * u * (1.0 - s2 / s1) / (kb_ic - h)) * tbi_ic

                _iupwb0[ic] = -g1 + n1b * u + n2b * v
                _iupwb[ic]  = -g1 * s2 + n1b * u * s1 + n2b * v / s1
                _idwnb[ic]  =  g2 * s2 - n1b * v * s1 - n2b * u / s1
                abs_b       = (1.0 - s2) - _iupwb0[ic] + _iupwb[ic] - _idwnb[ic]
                abs_b_sun   = (1.0 - om_ic) * ((1.0 - s2) + cf_ic / av_ic * (a1b + a2b))
                _iabsb_sun[ic] = abs_b_sun
                _iabsb_sha[ic] = abs_b - abs_b_sun

                # Diffuse terms (unitd = 1)
                num1 = (u + v * albd) * s1
                n2d  = num1 / (den1 - den2)
                n1d  = -(1.0 + n2d * u) / v

                a1d = (  n1d * u * (1.0 - s2 * s1) / (kb_ic + h)
                        + n2d * v * (1.0 - s2 / s1) / (kb_ic - h)) * tbi_ic
                a2d = (- n1d * v * (1.0 - s2 * s1) / (kb_ic + h)
                        - n2d * u * (1.0 - s2 / s1) / (kb_ic - h)) * tbi_ic

                _iupwd0[ic] = n1d * u + n2d * v
                _iupwd[ic]  = n1d * u * s1 + n2d * v / s1
                _idwnd[ic]  = -n1d * v * s1 - n2d * u / s1
                abs_d       = 1.0 - _iupwd0[ic] + _iupwd[ic] - _idwnd[ic]
                abs_d_sun   = (1.0 - om_ic) * cf_ic / av_ic * (a1d + a2d)
                _iabsd_sun[ic] = abs_d_sun
                _iabsd_sha[ic] = abs_d - abs_d_sun

                # Update albedos for next layer
                albb = _iupwb0[ic]
                albd = _iupwd0[ic]

            # ----------------------------------------------------------
            # Top-to-bottom sweep — Fortran lines 536-570
            # ----------------------------------------------------------
            dir_ = float(_swskyb[ib])
            dif  = float(_swskyd[ib])

            for ic in range(_ntop, _nbot - 1, -1):
                _swbeam_p[ic, ib] = dir_
                _swdwn_p[ic, ib]  = dif
                _swupw_p[ic, ib]  = _iupwd0[ic] * dif + _iupwb0[ic] * dir_

                fs    = _fracsun[ic]
                dp_ic = _dpai[ic]
                _swleaf_p[ic, isun, ib] = (
                    (_iabsb_sun[ic] * dir_ + _iabsd_sun[ic] * dif) / (fs * dp_ic)
                )
                _swleaf_p[ic, isha, ib] = (
                    (_iabsb_sha[ic] * dir_ + _iabsd_sha[ic] * dif) / ((1.0 - fs) * dp_ic)
                )

                kb_ic   = _kb[ic]
                cf_ic   = _cf[ic]
                dif_new = dir_ * _idwnb[ic] + dif * _idwnd[ic]
                dir_    = dir_ * math.exp(-kb_ic * cf_ic * dp_ic)
                dif     = dif_new

            # Ground fluxes
            _swbeam_p[0, ib] = dir_
            _swdwn_p[0, ib]  = dif
            _swupw_p[0, ib]  = _alb_d[ib] * dif + _alb_b[ib] * dir_
            _swsoi_p[ib] = dir_ * (1.0 - _alb_b[ib]) + dif * (1.0 - _alb_d[ib])

            # Canopy albedo
            suminc = float(_swskyb[ib]) + float(_swskyd[ib])
            sumref = _iupwb0[_ntop] * float(_swskyb[ib]) + _iupwd0[_ntop] * float(_swskyd[ib])
            _albcan_p[ib] = sumref / suminc if suminc > 0.0 else 0.0

            # Sum vegetation absorption
            swveg_ib = 0.0; svs_ib = 0.0; svsh_ib = 0.0
            for ic in range(_nbot, _ntop + 1):
                fs    = _fracsun[ic]
                dp_ic = _dpai[ic]
                sun   = _swleaf_p[ic, isun, ib] * fs * dp_ic
                sha   = _swleaf_p[ic, isha, ib] * (1.0 - fs) * dp_ic
                swveg_ib += sun + sha
                svs_ib   += sun
                svsh_ib  += sha
            _swveg_p[ib]    = swveg_ib
            _swvegsun_p[ib] = svs_ib
            _swvegsha_p[ib] = svsh_ib

            # Conservation check
            sumabs = swveg_ib + _swsoi_p[ib]
            if abs(suminc - (sumabs + _albcan_p[ib] * suminc)) >= 1.0e-6:
                endrun(msg='ERROR: TwoStream: total solar radiation conservation error')

            # DEBUG: print SW diagnostics for first few calls
        # Bulk JAX write-back for this patch
        _sl_lev    = slice(0, _ncan + 1)
        _sl_canopy = slice(1, _ncan + 1)
        _sl_rad    = slice(1, numrad + 1)
        swbeam = swbeam.at[p, _sl_lev, _sl_rad].set(jnp.array(_swbeam_p[_sl_lev, _sl_rad]))
        swdwn  = swdwn.at[p,  _sl_lev, _sl_rad].set(jnp.array(_swdwn_p[_sl_lev,  _sl_rad]))
        swupw  = swupw.at[p,  _sl_lev, _sl_rad].set(jnp.array(_swupw_p[_sl_lev,  _sl_rad]))
        swsoi  = swsoi.at[p,  _sl_rad].set(jnp.array(_swsoi_p[_sl_rad]))
        albcan = albcan.at[p, _sl_rad].set(jnp.array(_albcan_p[_sl_rad]))
        swveg    = swveg.at[p,    _sl_rad].set(jnp.array(_swveg_p[_sl_rad]))
        swvegsun = swvegsun.at[p, _sl_rad].set(jnp.array(_swvegsun_p[_sl_rad]))
        swvegsha = swvegsha.at[p, _sl_rad].set(jnp.array(_swvegsha_p[_sl_rad]))
        swleaf = swleaf.at[p, _sl_canopy, isun, _sl_rad].set(
            jnp.array(_swleaf_p[_sl_canopy, isun, _sl_rad])
        )
        swleaf = swleaf.at[p, _sl_canopy, isha, _sl_rad].set(
            jnp.array(_swleaf_p[_sl_canopy, isha, _sl_rad])
        )

    return mlcanopy_inst._replace(
        swveg_canopy    = swveg,
        swvegsun_canopy = swvegsun,
        swvegsha_canopy = swvegsha,
        albcan_canopy   = albcan,
        swsoi_soil      = swsoi,
        swleaf_leaf     = swleaf,
        swupw_profile   = swupw,
        swdwn_profile   = swdwn,
        swbeam_profile  = swbeam,
    )