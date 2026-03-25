"""
JAX translation of MLPlantHydraulicsMod Fortran module.

Calculate plant hydraulics for the multilayer canopy model.
Three public routines:

- :func:`PlantResistance`: whole-plant leaf-specific conductance.
- :func:`SoilResistance`: soil hydraulic resistance and fractional
  water uptake per soil layer.
- :func:`LeafWaterPotential`: leaf water potential via analytical
  ODE integration.

Original Fortran module: MLPlantHydraulicsMod
Fortran lines 1-200
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from clm_src_main.abortutils import endrun                               # noqa: F401
from clm_src_main.clm_varctl import iulog                               # noqa: F401
from clm_src_main.clm_varcon import rpi as pi, denh2o, grav             # noqa: F401
from clm_src_main.clm_varpar import nlevsoi                             # noqa: F401
from clm_src_main.ColumnType import col                                  # noqa: F401
from clm_src_main.PatchType import patch                                 # noqa: F401
from multilayer_canopy.MLpftconMod import MLpftcon                           # noqa: F401
from multilayer_canopy.MLclm_varcon import mmh2o                             # noqa: F401
from multilayer_canopy.MLclm_varctl import dtime_ml                          # noqa: F401
from clm_src_biogeophys.SoilStateType import soilstate_type                   # noqa: F401
from clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type         # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type               # noqa: F401


# ---------------------------------------------------------------------------
# Public: whole-plant leaf-specific conductance
# ---------------------------------------------------------------------------

def PlantResistance(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate whole-plant leaf-specific conductance (soil-to-leaf).

    Mirrors Fortran subroutine ``PlantResistance`` (lines 23-75).

    Reference: Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A21)-(A22).

    For each layer ``ic`` with ``dpai > 0``:

    .. code-block:: none

        rplant = 1 / gplant_SPA(pft)        [MPa.s.m2/mmol H2O]
        lsc(p,ic) = 1 / (rsoil(p) + rplant) [mmol H2O/m2/s/MPa]

    The commented-out height-dependent alternative
    (``rplant = zs / gplant_SPA``) is preserved as a comment.
    Layers with ``dpai == 0`` receive ``lsc = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``lsc_profile`` is updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    gplant_SPA = MLpftcon.gplant_SPA

    ncan  = mlcanopy_inst.ncan_canopy
    rsoil = mlcanopy_inst.rsoil_soil
    dpai  = mlcanopy_inst.dpai_profile
    zs    = mlcanopy_inst.zs_profile
    lsc   = mlcanopy_inst.lsc_profile

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])

        for ic in range(1, int(ncan[p]) + 1):          # Fortran: do ic = 1, ncan(p)
            if float(dpai[p, ic]) > 0.0:

                # Aboveground plant resistance (MPa.s.m2/mmol H2O) — Fortran lines 57-60
                # rplant = zs(p,ic) / gplant_SPA(pft)  # conductivity form (commented out)
                rplant = 1.0 / float(gplant_SPA[pft])   # conductance form

                # Leaf-specific conductance soil-to-leaf — Fortran line 63
                lsc = lsc.at[p, ic].set(1.0 / (float(rsoil[p]) + rplant))

            else:
                lsc = lsc.at[p, ic].set(0.0)           # Fortran line 67

    return mlcanopy_inst._replace(lsc_profile = lsc)


# ---------------------------------------------------------------------------
# Public: soil hydraulic resistance and fractional water uptake
# ---------------------------------------------------------------------------

def SoilResistance(
    num_filter: int,
    filter_patch: Sequence[int],
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate soil hydraulic resistance and fractional water uptake
    from each soil layer.

    Mirrors Fortran subroutine ``SoilResistance`` (lines 77-165).

    Reference: Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A23)-(A28).

    Per-layer calculations (Fortran lines 118-146):

    .. code-block:: none

        hk [mmol/m/s/MPa] = hk_l [mm/s] * (1e-3/head) * denh2o/mmh2o * 1000
        smp_mpa [MPa]      = smp_l [mm] * 1e-3 * head
        root_biomass_density [g/m3] = root_biomass * rootfr / dz    (≥ 1e-10)
        root_length_density  [m/m3] = root_biomass_density / (root_density * pi*r^2)
        root_dist [m]               = sqrt(1 / (root_length_density * pi))
        soilr1 = log(root_dist/r) / (2*pi * rld * dz * hk)         (A23)
        soilr2 = root_resist / (root_biomass_density * dz)          (A24)
        soilr  = soilr1 + soilr2
        rsoil  += 1/soilr    (sum conductances across layers)
        evap[j] = max((smp_mpa - minlwp_SPA) / soilr, 0)           (A26)
        evap[j] = 0 if frozen

    After the layer loop (Fortran lines 149-168):

    .. code-block:: none

        rsoil(p) = lai(p) / rsoil(p)           (A25, resistance form)
        psis(p)  = sum(smp_mpa*evap) / totevap  (A27-A28)
        soil_et_loss(p,j) = evap(j) / totevap

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        soilstate_inst: Soil state container (read-only).
        waterstatebulk_inst: Bulk water state container (read-only).
        mlcanopy_inst: Canopy container; ``psis_soil``,
            ``rsoil_soil``, and ``soil_et_loss_soil`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    minlwp_SPA: float = -2.0    # Fortran local parameter (line 95)

    head: float = denh2o * grav * 1.0e-6    # MPa/m

    root_radius_SPA  = MLpftcon.root_radius_SPA
    root_density_SPA = MLpftcon.root_density_SPA
    root_resist_SPA  = MLpftcon.root_resist_SPA

    dz_col      = col.dz
    nbedrock    = col.nbedrock
    smp_l       = soilstate_inst.smp_l_col
    hk_l        = soilstate_inst.hk_l_col
    rootfr      = soilstate_inst.rootfr_patch
    h2osoi_ice  = waterstatebulk_inst.h2osoi_ice_col

    lai          = mlcanopy_inst.lai_canopy
    root_biomass = mlcanopy_inst.root_biomass_canopy
    psis         = mlcanopy_inst.psis_soil
    rsoil        = mlcanopy_inst.rsoil_soil
    soil_et_loss = mlcanopy_inst.soil_et_loss_soil

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])
        c = int(patch.column[p])
        pft = int(patch.itype[p])

        nlayers = int(nbedrock[c])                     # Fortran: nlayers = nbedrock(c)

        # Root cross-sectional area (m2 root) — Fortran line 111
        rr = float(root_radius_SPA[pft])
        root_cross_sec_area = pi * rr * rr

        rsoil_sum: float = 0.0
        totevap:   float = 0.0
        smp_mpa  = [0.0] * (nlayers + 1)     # 1-based local
        evap_arr = [0.0] * (nlayers + 1)     # 1-based local

        for j in range(1, nlayers + 1):                # Fortran: do j = 1, nlayers

            # Hydraulic conductivity and matric potential — Fortran lines 116-118
            hk = float(hk_l[c, j]) * (1.0e-3 / head)         # mm/s → m2/s/MPa
            hk = hk * denh2o / mmh2o * 1000.0                 # → mmol/m/s/MPa
            smp_mpa[j] = float(smp_l[c, j]) * 1.0e-3 * head  # mm → MPa

            # Root biomass density (g biomass/m3 soil) — Fortran lines 120-122
            dz_j = float(dz_col[c, j])
            rbd  = float(root_biomass[p]) * float(rootfr[p, j]) / dz_j
            rbd  = max(rbd, 1.0e-10)                           # Fortran: max(..., 1e-10)

            # Root length density (m root/m3 soil) — Fortran line 125
            rld = rbd / (float(root_density_SPA[pft]) * root_cross_sec_area)

            # Mean distance between roots (m) — Fortran line 128
            root_dist = math.sqrt(1.0 / (rld * pi))

            # Soil-to-root resistance (A23) — Fortran line 131
            soilr1 = math.log(root_dist / rr) / (2.0 * pi * rld * dz_j * hk)

            # Root-to-stem resistance (A24) — Fortran line 134
            soilr2 = float(root_resist_SPA[pft]) / (rbd * dz_j)

            # Belowground resistance — Fortran line 137
            soilr_j = soilr1 + soilr2

            # Sum conductances — Fortran line 141
            rsoil_sum += 1.0 / soilr_j

            # Maximum transpiration per layer (A26) — Fortran lines 145-148
            evap_j = (smp_mpa[j] - minlwp_SPA) / soilr_j
            evap_j = max(evap_j, 0.0)
            if float(h2osoi_ice[c, j]) > 0.0:
                evap_j = 0.0
            evap_arr[j] = evap_j
            totevap += evap_j

        # Total belowground resistance (A25) — Fortran line 151
        rsoil = rsoil.at[p].set(float(lai[p]) / rsoil_sum)

        # Weighted soil water potential and fractional uptake — Fortran lines 153-168
        psis_p: float = 0.0
        for j in range(1, nlayers + 1):
            psis_p += smp_mpa[j] * evap_arr[j]
            if totevap > 0.0:
                soil_et_loss = soil_et_loss.at[p, j].set(evap_arr[j] / totevap)
            else:
                soil_et_loss = soil_et_loss.at[p, j].set(1.0 / nlayers)

        psis_p = psis_p / totevap if totevap > 0.0 else minlwp_SPA
        psis = psis.at[p].set(psis_p)

    return mlcanopy_inst._replace(
        psis_soil         = psis,
        rsoil_soil        = rsoil,
        soil_et_loss_soil = soil_et_loss,
    )


# ---------------------------------------------------------------------------
# Public: leaf water potential
# ---------------------------------------------------------------------------

def LeafWaterPotential(
    num_filter: int,
    filter_patch: Sequence[int],
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf water potential by analytically integrating the
    first-order ODE over one multilayer canopy timestep.

    Mirrors Fortran subroutine ``LeafWaterPotential`` (lines 167-200).

    Reference: Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222,
    doi:10.5194/gmd-7-2193-2014, eqs. (A19)-(A20).

    The ODE ``dy/dt = (a - y) / b`` has the analytical solution over
    a full timestep ``dtime``:

    .. code-block:: none

        a  = psis(p) - head*zs(p,ic) - 1000*trleaf(p,ic,il)/lsc(p,ic)
        b  = capac_SPA(pft) / lsc(p,ic)
        dy = (a - y0) * (1 - exp(-dtime/b))
        lwp(p,ic,il) = y0 + dy

    where ``y0 = lwp_bef(p,ic,il)`` is the leaf water potential from
    the previous timestep.  Layers with ``dpai == 0`` receive
    ``lwp = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; ``lwp_leaf`` is updated for
            leaf type ``il``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    head: float = denh2o * grav * 1.0e-6    # MPa/m

    dtime = float(dtime_ml)                 # Multilayer canopy timestep (s)

    capac_SPA = MLpftcon.capac_SPA

    ncan    = mlcanopy_inst.ncan_canopy
    psis    = mlcanopy_inst.psis_soil
    dpai    = mlcanopy_inst.dpai_profile
    zs      = mlcanopy_inst.zs_profile
    lsc     = mlcanopy_inst.lsc_profile
    trleaf  = mlcanopy_inst.trleaf_leaf
    lwp_bef = mlcanopy_inst.lwp_bef_leaf
    lwp     = mlcanopy_inst.lwp_leaf

    for fp in range(1, num_filter + 1):                # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])
        _ncan = int(ncan[p])

        # Pre-extract patch scalars and layer profiles as numpy (one sync each)
        _psis_p  = float(psis[p])
        _capac_p = float(capac_SPA[pft])
        _dpai    = np.asarray(dpai[p])
        _zs      = np.asarray(zs[p])
        _lsc     = np.asarray(lsc[p])
        _trleaf  = np.asarray(trleaf[p, :, il])
        _lwp_bef = np.asarray(lwp_bef[p, :, il])

        ics      = np.arange(1, _ncan + 1)
        dpai_v   = _dpai[ics]
        has_pai  = dpai_v > 0.0

        lsc_safe = np.where(has_pai, _lsc[ics], 1.0)   # avoid div-by-zero
        a_v      = np.where(
            has_pai,
            _psis_p - head * _zs[ics] - 1000.0 * _trleaf[ics] / lsc_safe,
            0.0,
        )
        b_v      = np.where(has_pai, _capac_p / lsc_safe, 1.0)
        y0_v     = _lwp_bef[ics]
        dy_v     = np.where(has_pai, (a_v - y0_v) * (1.0 - np.exp(-dtime / b_v)), 0.0)

        _lwp_new      = np.zeros(_ncan + 2)
        _lwp_new[ics] = np.where(has_pai, y0_v + dy_v, 0.0)

        _sl = slice(1, _ncan + 1)
        lwp = lwp.at[p, _sl, il].set(jnp.array(_lwp_new[_sl]))

    return mlcanopy_inst._replace(lwp_leaf = lwp)