"""
Smoke test for jax.grad(clm_ml_forward).

Reuses the full main.py initialization flow, runs 1 timestep,
then tests the differentiable forward pass and jax.grad.
"""
import os
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Monkey-patch sys.argv so main.py sees the namelist
namelist_file = str(Path(__file__).parent / 'nl.CHATS7.1day')
sys.argv = ['test_grad.py', namelist_file]

# Import and call main() but stop after 1 timestep by patching ntimes
import offline_executable.main as _main_mod

# We need to run main() but intercept after 1 timestep.
# Easiest: just replicate the init portion from main().
nml = _main_mod.read_namelist(namelist_file)
params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

tower_name = str(params.get("tower_name", "CHATS7"))
start_ymd  = int(params.get("start_ymd", 20070501))
iyear      = start_ymd // 10000
imonth     = (start_ymd // 100) % 100

fin_tower       = _main_mod._resolve_path(str(params.get("fin_tower", "")))
fin_clm         = _main_mod._resolve_path(str(params.get("fin_clm", "")))
fin_soil_adjust = _main_mod._resolve_path(str(params.get("fin_soil_adjust", "")))
dirout_raw      = str(params.get("dirout", "")).strip()
dirout          = _main_mod._resolve_path(dirout_raw) if dirout_raw else ""

fin1, fin2 = fin_tower, fin_clm

from offline_driver import controlMod
controlMod.tower_site = tower_name
controlMod.iyear = iyear
controlMod.imonth = imonth

from offline_driver import TowerDataMod, clmSoilOptionMod
from multilayer_canopy import MLclm_varctl
from clm_src_utils import clm_time_manager
from clm_src_utils.clm_time_manager import get_curr_calday

tower_num = 0
for i in range(1, int(TowerDataMod.ntower) + 1):
    if tower_name == str(TowerDataMod.tower_id[i]):
        tower_num = i
        break
TowerDataMod.tower_num = tower_num

clmSoilOptionMod.clm_phys = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
clmSoilOptionMod.nlev_soil_adjust = int(params.get("nlev_soil_adjust", 0))
clmSoilOptionMod.fin_soil_adjust = fin_soil_adjust
MLclm_varctl.met_type    = int(params.get("met_type", MLclm_varctl.met_type))
MLclm_varctl.dpai_min    = float(params.get("dpai_min", MLclm_varctl.dpai_min))
MLclm_varctl.pftcon_val  = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))

clm_time_manager.start_date_ymd = start_ymd
clm_time_manager.start_date_tod = int(params.get("start_tod", 0))
dtstep_sec = int(TowerDataMod.tower_time[tower_num]) * 60
clm_time_manager.dtstep = dtstep_sec

bounds = _main_mod.build_bounds(nml)

ntimes = 2

from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance
from clm_src_main import filterMod as _filterMod

InitializeRealize(bounds)

from clm_src_main import clm_instMod

_new_filter = _filterMod.setFilters(_filterMod.filter)
_filterMod.filter = _new_filter
import clm_src_main.clm_driver as _clm_driver_mod
_clm_driver_mod.filter = _new_filter

from offline_driver.CLMml_driver import init_acclim, TowerVeg, SoilInit
from offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext
from clm_share.shr_orb_mod import shr_orb_params
from clm_src_utils import clm_varorb

_eccen, _obliq, _mvelp, _obliqr, _lambm0, _mvelpp = shr_orb_params(iyear)
clm_varorb.eccen  = _eccen
clm_varorb.obliqr = _obliqr
clm_varorb.lambm0 = _lambm0
clm_varorb.mvelpp = _mvelpp

(clm_instMod.atm2lnd_inst,
 clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.temperature_inst,
 clm_instMod.frictionvel_inst,
 clm_instMod.mlcanopy_inst) = init_acclim(
    fin1, tower_num, ntimes,
    bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.temperature_inst,
    clm_instMod.frictionvel_inst,
    clm_instMod.mlcanopy_inst,
)

(clm_instMod.canopystate_inst,
 clm_instMod.mlcanopy_inst) = TowerVeg(
    tower_num, bounds.begp, bounds.endp,
    clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst,
)

# Soil init time index
clm_start_ymd = int(params.get("clm_start_ymd", start_ymd))
clm_start_tod = int(params.get("clm_start_tod", 0))
run_start_ymd = clm_time_manager.start_date_ymd
run_start_tod = clm_time_manager.start_date_tod
clm_time_manager.start_date_ymd = clm_start_ymd
clm_time_manager.start_date_tod = clm_start_tod
clm_time_manager.itim = 1
start_calday_clm = get_curr_calday(offset=0)
clm_time_manager.start_date_ymd = run_start_ymd
clm_time_manager.start_date_tod = run_start_tod
clm_time_manager.itim = 1
curr_calday = get_curr_calday(offset=0)
soil_init_time_indx = round((curr_calday - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.waterstatebulk_inst,
 clm_instMod.temperature_inst) = SoilInit(
    fin2, soil_init_time_indx,
    bounds.begc, bounds.endc,
    clm_instMod.soilstate_inst,
    clm_instMod.waterstatebulk_inst,
    clm_instMod.temperature_inst,
)

print("Initialization complete.", flush=True)

os.makedirs(dirout, exist_ok=True)

# Load met data for timestep 1 (required before ModelAdvance)
met_type = MLclm_varctl.met_type
clm_time_manager.itim = 1
from clm_src_utils.clm_time_manager import get_curr_date
yr, mon, day, _ = get_curr_date()
curr_calday_ts1 = get_curr_calday(offset=0)
time_indx = round((curr_calday_ts1 - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.atm2lnd_inst,
 clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin1, 1, tower_num,
    bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)

if met_type == 3:
    clm_instMod.mlcanopy_inst = TowerMetNext(
        fin1, min(2, ntimes),
        bounds.begp, bounds.endp,
        clm_instMod.mlcanopy_inst,
    )

# Run 1 timestep
ModelAdvance(bounds, time_indx, fin1, fin2)

print("\nFirst timestep complete. Testing differentiable forward pass.\n")

# --- Differentiable forward pass ---
from multilayer_canopy.MLCanopyFluxesMod import make_clm_ml_forward
from multilayer_canopy import MLclm_varctl
from clm_src_utils import clm_time_manager as _ctm

# Reduce computation graph size for NaN-checking:
# Full run uses num_ml_steps=6 × (nrk_steps+1)=5 = 30 iterations.
# jax.grad of 30 iterations OOMs (103 GB observed).
# Use Euler + 1 sub-step → 1×1 = 1 iteration for NaN testing.
_orig_rk    = MLclm_varctl.runge_kutta_type
_orig_dtime = MLclm_varctl.dtime_ml
MLclm_varctl.runge_kutta_type = 10               # Euler: nrk_steps=0
MLclm_varctl.dtime_ml         = float(_ctm.dtstep)  # num_ml_steps=1
print(f"NaN test mode: Euler (nrk=0), 1 sub-step (dtime_ml={MLclm_varctl.dtime_ml}s)")

mlcanopy_inst = clm_instMod.mlcanopy_inst
p = bounds.begp
print(f"p={p}, ncan={int(mlcanopy_inst.ncan_canopy[p])}")

filt = _clm_driver_mod.filter
num_evp = filt.num_exposedvegp
evp_list = [int(v) for v in filt.exposedvegp[:num_evp]]
print(f"num_exposedvegp={num_evp}, filter={evp_list}")

forward_fn = make_clm_ml_forward(
    mlcanopy_inst_template=mlcanopy_inst,
    bounds=bounds,
    num_exposedvegp=num_evp,
    filter_exposedvegp=evp_list,
    atm2lnd_inst=clm_instMod.atm2lnd_inst,
    canopystate_inst=clm_instMod.canopystate_inst,
    soilstate_inst=clm_instMod.soilstate_inst,
    temperature_inst=clm_instMod.temperature_inst,
    waterstatebulk_inst=clm_instMod.waterstatebulk_inst,
    waterfluxbulk_inst=clm_instMod.waterfluxbulk_inst,
    energyflux_inst=clm_instMod.energyflux_inst,
    frictionvel_inst=clm_instMod.frictionvel_inst,
    surfalb_inst=clm_instMod.surfalb_inst,
    solarabs_inst=clm_instMod.solarabs_inst,
    wateratm2lndbulk_inst=clm_instMod.wateratm2lndbulk_inst,
    waterdiagnosticbulk_inst=clm_instMod.waterdiagnosticbulk_inst,
)

print("\nRunning differentiable forward pass...")
with jax.disable_jit():
    loss = forward_fn(mlcanopy_inst)
print(f"Forward loss = {loss}")

print("\nComputing jax.grad (eager, jax.disable_jit to avoid slow FPS recompilation)...")
try:
    grad_fn = jax.grad(forward_fn, allow_int=True)
    # Use disable_jit so _implicit_fps_jit runs as plain Python:
    # this avoids the ~5-min XLA compilation of the backward pass through JIT.
    with jax.disable_jit():
        grads = grad_fn(mlcanopy_inst)
    print("jax.grad completed successfully!\n")

    for field_name in ['tair_profile', 'eair_profile', 'tleaf_leaf', 'tg_soil']:
        grad_val = getattr(grads, field_name)
        if grad_val is not None:
            print(f"  grad({field_name}): shape={grad_val.shape}, "
                  f"max_abs={float(jnp.max(jnp.abs(grad_val))):.6e}, "
                  f"mean_abs={float(jnp.mean(jnp.abs(grad_val))):.6e}")
except Exception as e:
    print(f"\njax.grad FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Restore original settings
MLclm_varctl.runge_kutta_type = _orig_rk
MLclm_varctl.dtime_ml         = _orig_dtime

print("\nAll tests passed!")
