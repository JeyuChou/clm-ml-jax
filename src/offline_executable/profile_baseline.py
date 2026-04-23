"""
Baseline profiler for MLCanopyFluxes.

Runs 1 CLM timestep (same init as test_grad.py) and reports wall-clock
timing for the top-level driver and major sub-functions.

Usage (from project root):
    python src/offline_executable/profile_baseline.py
"""
import os
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -----------------------------------------------------------------------
# Initialization (mirrors test_grad.py)
# -----------------------------------------------------------------------
namelist_file = str(Path(__file__).parent / 'nl.CHATS7.1day')
sys.argv = ['profile_baseline.py', namelist_file]

import offline_executable.main as _main_mod

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

# -----------------------------------------------------------------------
# Run 1 warmup timestep (allows lazy init to complete)
# -----------------------------------------------------------------------
print("\nRunning warmup timestep (for JIT compilation)...", flush=True)
t0 = time.perf_counter()
ModelAdvance(bounds, time_indx, fin1, fin2)
jax.effects_barrier()
t_warmup = time.perf_counter() - t0
print(f"  Warmup (includes JIT compile): {t_warmup:.3f} s", flush=True)

# -----------------------------------------------------------------------
# Profile MLCanopyFluxes directly on second call (post-compile)
# -----------------------------------------------------------------------
print("\nProfiling MLCanopyFluxes (post-JIT)...", flush=True)

from multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes, make_clm_ml_forward
from multilayer_canopy import MLclm_varctl as _ctl

mlcanopy_inst = clm_instMod.mlcanopy_inst

filt = _clm_driver_mod.filter
num_evp = filt.num_exposedvegp
evp_list = [int(v) for v in filt.exposedvegp[:num_evp]]

# Time full MLCanopyFluxes call (non-diff mode, with diagnostics)
N_ITERS = 3
times_full = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    result = MLCanopyFluxes(
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
        mlcanopy_inst=mlcanopy_inst,
        wateratm2lndbulk_inst=clm_instMod.wateratm2lndbulk_inst,
        waterdiagnosticbulk_inst=clm_instMod.waterdiagnosticbulk_inst,
        grid=None,
    )
    jax.effects_barrier()
    times_full.append(time.perf_counter() - t0)

# Time the differentiable forward pass (diff mode, no diagnostics)
_p = evp_list[0]
from multilayer_canopy.MLclm_varctl import GridInfo
grid = GridInfo(
    p=_p,
    ncan=int(mlcanopy_inst.ncan_canopy[_p]),
    ntop=int(mlcanopy_inst.ntop_canopy[_p]),
    nbot=int(mlcanopy_inst.nbot_canopy[_p]),
)

times_diff = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    result_diff = MLCanopyFluxes(
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
        mlcanopy_inst=mlcanopy_inst,
        wateratm2lndbulk_inst=clm_instMod.wateratm2lndbulk_inst,
        waterdiagnosticbulk_inst=clm_instMod.waterdiagnosticbulk_inst,
        grid=grid,
    )
    jax.effects_barrier()
    times_diff.append(time.perf_counter() - t0)

# Time differentiable forward + grad
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

grad_fn = jax.jit(jax.grad(forward_fn, allow_int=True))

print("  Compiling grad_fn (first call)...", flush=True)
t0 = time.perf_counter()
grads = grad_fn(mlcanopy_inst)
jax.effects_barrier()
t_grad_compile = time.perf_counter() - t0
print(f"  grad_fn first call (compile): {t_grad_compile:.3f} s", flush=True)

times_grad = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    grads = grad_fn(mlcanopy_inst)
    jax.effects_barrier()
    times_grad.append(time.perf_counter() - t0)

# -----------------------------------------------------------------------
# Memory stats (if available)
# -----------------------------------------------------------------------
devices = jax.devices()
mem_info = ""
try:
    stats = devices[0].memory_stats()
    if stats:
        used_mb = stats.get('bytes_in_use', 0) / 1024**2
        peak_mb = stats.get('peak_bytes_in_use', 0) / 1024**2
        mem_info = f"  Device memory in use: {used_mb:.1f} MB  (peak {peak_mb:.1f} MB)"
    else:
        mem_info = "  Device memory stats: not available"
except Exception as e:
    mem_info = f"  Device memory stats: {e}"

# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("PROFILING REPORT — CHATS7 1 CLM timestep (baseline)")
print("="*65)
print(f"  Device: {devices[0]}")
print(f"  JAX version: {jax.__version__}")
print(f"  num_ml_steps: {int(dtstep_sec / MLclm_varctl.dtime_ml)}")
print(f"  nstep_rk inner: {MLclm_varctl.runge_kutta_type // 10 + 1}")
print()
print(f"  Warmup (JIT compile + 1 ModelAdvance): {t_warmup:.3f} s")
print()
print(f"  MLCanopyFluxes (non-diff, {N_ITERS} iters):")
print(f"    min={min(times_full):.3f} s  mean={sum(times_full)/N_ITERS:.3f} s  max={max(times_full):.3f} s")
print()
print(f"  MLCanopyFluxes (diff mode, {N_ITERS} iters):")
print(f"    min={min(times_diff):.3f} s  mean={sum(times_diff)/N_ITERS:.3f} s  max={max(times_diff):.3f} s")
print()
print(f"  jax.grad(forward_fn) first call (compile): {t_grad_compile:.3f} s")
print(f"  jax.grad(forward_fn) post-compile ({N_ITERS} iters):")
print(f"    min={min(times_grad):.3f} s  mean={sum(times_grad)/N_ITERS:.3f} s  max={max(times_grad):.3f} s")
print()
print(mem_info)
print("="*65)

# Save results to file
results_path = Path(__file__).parent / 'profile_results.txt'
with open(results_path, 'w') as f:
    f.write("PROFILING REPORT — CHATS7 1 CLM timestep (baseline)\n")
    f.write("="*65 + "\n")
    f.write(f"Device: {devices[0]}\n")
    f.write(f"JAX version: {jax.__version__}\n")
    f.write(f"num_ml_steps: {int(dtstep_sec / MLclm_varctl.dtime_ml)}\n")
    f.write(f"nstep_rk inner: {MLclm_varctl.runge_kutta_type // 10 + 1}\n\n")
    f.write(f"Warmup (JIT compile + 1 ModelAdvance): {t_warmup:.3f} s\n\n")
    f.write(f"MLCanopyFluxes (non-diff, {N_ITERS} iters):\n")
    f.write(f"  min={min(times_full):.3f} s  mean={sum(times_full)/N_ITERS:.3f} s  max={max(times_full):.3f} s\n\n")
    f.write(f"MLCanopyFluxes (diff mode, {N_ITERS} iters):\n")
    f.write(f"  min={min(times_diff):.3f} s  mean={sum(times_diff)/N_ITERS:.3f} s  max={max(times_diff):.3f} s\n\n")
    f.write(f"jax.grad first call (compile): {t_grad_compile:.3f} s\n")
    f.write(f"jax.grad post-compile ({N_ITERS} iters):\n")
    f.write(f"  min={min(times_grad):.3f} s  mean={sum(times_grad)/N_ITERS:.3f} s  max={max(times_grad):.3f} s\n\n")
    f.write(f"{mem_info}\n")

print(f"\nResults saved to: {results_path}")
