"""Debug script to trace physics values through one timestep."""
import jax
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clm_src_main.decompMod import bounds_type
from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance
from offline_driver import controlMod, TowerDataMod
from clm_src_utils import clm_time_manager

controlMod.tower_site = 'CHATS7'
controlMod.iyear = 2007; controlMod.imonth = 5
for i in range(1, 16):
    if TowerDataMod.tower_id[i] == 'CHATS7':
        TowerDataMod.tower_num = i; break

clm_time_manager.start_date_ymd = 20070501
clm_time_manager.start_date_tod = 0
clm_time_manager.dtstep = int(TowerDataMod.tower_time[TowerDataMod.tower_num]) * 60

bounds = bounds_type(begg=0, endg=0, begl=0, endl=0, begc=0, endc=0, begp=0, endp=0)
from clm_src_main.filterMod import setFilters, filter as clm_filter
InitializeRealize(bounds)
from clm_src_main import clm_instMod
setFilters(clm_filter)

# Load met data for first timestep
fin1 = str(Path(__file__).parent.parent / "input_files" / "CHATS7_2007-05.nc")
from offline_driver.TowerMetMod import TowerMetCurr
from offline_driver.CLMml_driver import TowerVeg, init_acclim, SoilInit
fin2 = str(Path(__file__).parent.parent / "input_files" / "clm_45_CHATS7_2007-05.nc")

# TowerVeg: sets patch.itype, LAI, SAI, htop, etc.
(clm_instMod.canopystate_inst,
 clm_instMod.mlcanopy_inst) = TowerVeg(
    TowerDataMod.tower_num, bounds.begp, bounds.endp,
    clm_instMod.canopystate_inst,
    clm_instMod.mlcanopy_inst,
)

print("After TowerVeg:")
p = 0
print(f"  elai_patch[{p}]:", float(clm_instMod.canopystate_inst.elai_patch[p]))
print(f"  esai_patch[{p}]:", float(clm_instMod.canopystate_inst.esai_patch[p]))
print(f"  mlcanopy lai_canopy[{p}]:", float(clm_instMod.mlcanopy_inst.lai_canopy[p]))

# TowerMetCurr for timestep 1
(clm_instMod.atm2lnd_inst,
 clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin1, 1, TowerDataMod.tower_num, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)

print("\nAfter TowerMetCurr (timestep 1):")
c = 0
print(f"  forc_t_col[{c}]:", float(clm_instMod.atm2lnd_inst.forc_t_downscaled_col[c]))
print(f"  forc_u_grc[0]:", float(clm_instMod.atm2lnd_inst.forc_u_grc[0]))
print(f"  forc_solad_col[{c},0]:", float(clm_instMod.atm2lnd_inst.forc_solad_downscaled_col[c, 0]))
print(f"  forc_hgt_u_patch[{p}]:", float(clm_instMod.frictionvel_inst.forc_hgt_u_patch[p]))

# Run one step
clm_time_manager.itim = 1
ModelAdvance(bounds)

print("\nAfter ModelAdvance (timestep 1):")
mlcan = clm_instMod.mlcanopy_inst
from clm_src_main.clm_varcon import spval
print(f"  rnet_canopy[{p}]:", float(mlcan.rnet_canopy[p]))
print(f"  shflx_canopy[{p}]:", float(mlcan.shflx_canopy[p]))
print(f"  lhflx_canopy[{p}]:", float(mlcan.lhflx_canopy[p]))
print(f"  tleaf_leaf[{p},0,0]:", float(mlcan.tleaf_leaf[p, 0, 0]))
print(f"  tair_profile[{p},0]:", float(mlcan.tair_profile[p, 0]) if hasattr(mlcan, 'tair_profile') else "N/A")
is_spval = abs(float(mlcan.rnet_canopy[p]) - spval) < 1e30
print(f"  rnet is spval: {is_spval}")
