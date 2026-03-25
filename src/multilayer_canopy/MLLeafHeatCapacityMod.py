"""
JAX translation of MLLeafHeatCapacityMod Fortran module.

Calculate leaf heat capacity for each canopy layer.

Original Fortran module: MLLeafHeatCapacityMod
Fortran lines 1-75
"""

from __future__ import annotations

from typing import Sequence

from clm_src_main.clm_varcon import cpliq                    # noqa: F401
from clm_src_main.PatchType import patch                     # noqa: F401
from clm_src_main.pftconMod import pftcon                    # noqa: F401
from multilayer_canopy.MLclm_varcon import cpbio, fcarbon, fwater  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type    # noqa: F401


def LeafHeatCapacity(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf heat capacity for each canopy layer.

    Mirrors Fortran subroutine ``LeafHeatCapacity`` (lines 22-75).

    Reference: Bonan et al. (2018) *Geosci. Model Dev.*, 11, 1467-1496,
    doi:10.5194/gmd-11-1467-2018, eq. (A29).

    The derivation converts specific leaf area (m2/gC) to leaf mass
    per area and then accounts for the water fraction of fresh biomass
    (Fortran lines 55-65):

    .. code-block:: none

        lma          = 1/slatop * 0.001          [kg C / m2 leaf]
        dry_weight   = lma / fcarbon             [kg DM / m2 leaf]
        fresh_weight = dry_weight / (1 - fwater) [kg FM / m2 leaf]
        leaf_water   = fwater * fresh_weight     [kg H2O / m2 leaf]
        cpleaf       = cpbio * dry_weight + cpliq * leaf_water  [J/K/m2 leaf]

    Layers with ``dpai == 0`` receive ``cpleaf = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; ``cpleaf_profile`` is updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    slatop = pftcon.slatop
    ncan   = mlcanopy_inst.ncan_canopy
    dpai   = mlcanopy_inst.dpai_profile
    cpleaf = mlcanopy_inst.cpleaf_profile

    for fp in range(1, num_filter + 1):            # Fortran: do fp = 1, num_filter
        p   = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])

        for ic in range(1, int(ncan[p]) + 1):      # Fortran: do ic = 1, ncan(p)

            if float(dpai[p, ic]) > 0.0:           # Fortran lines 56-64

                # Leaf carbon mass per area: m2/gC → kg C/m2 — Fortran line 57
                lma = 1.0 / float(slatop[pft]) * 0.001

                # Leaf dry mass per area: kg C/m2 → kg DM/m2 — Fortran line 58
                dry_weight = lma / fcarbon

                # Leaf fresh mass per area: kg DM/m2 → kg FM/m2 — Fortran line 59
                fresh_weight = dry_weight / (1.0 - fwater)

                # Leaf water content: kg H2O/m2 leaf — Fortran line 60
                leaf_water = fwater * fresh_weight

                # Heat capacity: J/K/m2 leaf — Fortran line 61
                cpleaf = cpleaf.at[p, ic].set(cpbio * dry_weight + cpliq * leaf_water)

            else:
                cpleaf = cpleaf.at[p, ic].set(0.0)  # Fortran line 63

    return mlcanopy_inst._replace(cpleaf_profile = cpleaf)