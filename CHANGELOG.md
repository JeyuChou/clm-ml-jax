# Changelog

## 2026-04-03 — GPU benchmark baseline (session 8)

**Status:** Script created and run. Results documented. Critical performance issue identified.

### Script: `diags/benchmark_gpu.py`

Times `MLCanopyFluxes` / `ModelAdvance` on the CHATS7 site (V100S GPU).
Output: `diags/figures/benchmark_baseline.txt`.

### Key baseline numbers

| Metric | Time |
|---|---|
| Compile + first run (timestep 1, num_mlcan=1) | **1987 s** (33 min) |
| Steady-state (timestep 2, num_mlcan=1) | **1942 s** (also recompiling!) |
| Compile + first run (num_mlcan=0, empty filter) | 1.4 s |
| Steady-state (num_mlcan=0, 5 calls) | 0.424 s/call (std 0.004 s) |
| Per-ml-substep estimate (num_mlcan=0) | 70.7 ms |

### Critical finding: recompilation every timestep

`_make_physics_step(calday, nstep)` factory was refactored in session 7 to
`_physics_step_fn(inst, calday_ml)` with `calday_ml` as explicit arg.
BUT: `calday_ml` is a Python float passed as a JIT-traced argument.
Each new float value is a different Python object → JAX JIT cache miss → full
XLA recompilation of `jit__physics_step_fn` every timestep (~1942 s/step).

Fix needed: Pass `calday_ml` as a JAX array (not Python float) so JAX traces
through its value rather than keying the cache on the Python object.

### Other issues documented

- **CUDA graph OOM**: Direct `MLCanopyFluxes` calls after `ModelAdvance` hit
  "RESOURCE_EXHAUSTED: 14 alive graphs" limit. Benchmark timing via
  sequential `ModelAdvance` calls avoids this.
- **D→H sync**: `int(mlcanopy_inst.ntop_canopy[p])` in LongwaveRadiation:317
  forces device→host transfer every sub-step.

### Next actions

- [ ] Fix calday_ml passing in MLCanopyFluxesMod: wrap float as `jnp.array(calday_ml)`
      before passing to JIT-compiled step function (or use `functools.partial` with
      concrete value to mark it static).
- [ ] After fix: re-run benchmark to measure true steady-state time.

---

## 2026-04-03 — JIT attempt for non-diff MLCanopyFluxes via GridInfo (session 7)

**Status:** Partial implementation. `calday_ml`-as-explicit-arg refactoring applied.
Full JIT not feasible due to XLA compilation OOM/timeout on available V100 hardware.

### What was done

Refactored `_make_physics_step(calday_ml, nstep)` factory pattern into a direct
`_physics_step_fn(inst, calday_ml)` function. Key changes:
- `calday_ml` is now an explicit argument instead of baked into a closure
- Both diff and non-diff modes call `_physics_step_fn` (diff mode wraps in `_step_for_checkpoint`)
- Code is simpler (no factory pattern)

### GridInfo-in-non-diff-mode: attempted and reverted

Tried constructing `GridInfo` before the physics loop in non-diff mode to activate JAX-traceable
code paths in all 6 physics modules, then JIT-compiling `_physics_step_fn`. This would fix all
the blockers documented in session 6.

**Why it failed:**
1. **XLA compilation OOM**: `jax.jit(_physics_step_fn)` crashed with
   `CUDA_ERROR_OUT_OF_MEMORY` during XLA CUBIN loading on the first sub-step.
   The XLA computation graph for 5 RK iterations × ~10 physics calls is too large
   for the 32GB V100.
2. **Eager mode with diff-mode paths is slower**: Using GridInfo activates the
   diff-mode code paths (e.g., `_HF2008_diff`, JAX array slicing instead of Python
   for-loops). In eager mode, these are significantly slower than the original
   non-diff paths that use Python scalars.
3. **GPU contention**: GPU 1 was full (31958/32768 MiB used), GPU 0 at 72%
   utilization from another process.

### Current state

- Non-diff mode: uses original code paths (Python for-loops, concrete Python ints)
- Diff mode: uses JAX-traceable paths + `jax.checkpoint`
- `_physics_step_fn` is architecturally ready for future JIT (calday_ml as explicit arg)
- To enable JIT in the future: would need dedicated GPU memory + possibly
  `jax.jit` with `backend_options` to limit XLA compilation memory

### Next steps for JIT

Option B (from session 6 CHANGELOG) remains the recommended path:
- Run JIT with exclusive GPU access (no other processes using the GPU)
- If OOM persists, use `XLA_FLAGS=--xla_gpu_enable_xla_runtime_executable=false` or
  reduce `nrk_steps` to test JIT feasibility with smaller graph
- Consider compiling on CPU and running on GPU (`jit(..., device=gpu_device)`)

---

## 2026-04-03 — JIT audit for non-diff MLCanopyFluxes (session 6)

**Status:** Audit completed. JIT at `_step_fn` granularity is **not feasible** without
multi-module refactoring. Documented blockers below. No code changes made.

### Task
Wrap the non-diff `_step_fn` (inside `MLCanopyFluxes`) with `@jax.jit` to eliminate
~30 separate GPU kernel dispatches per sub-step (30 sub-steps × ~24 dispatches = 720 per
CLM timestep).

### Blockers found (exhaustive)

JIT on a traced function fails when any code inside it calls `int()`, `float()`, or
`np.asarray()` on a JAX traced array.  The non-diff `_step_fn` calls these patterns
across **6 modules**:

#### 1. `MLCanopyTurbulenceMod._HF2008` (biggest blocker)
The non-diff turbulence driver extracts ~15 Python scalars from `mlcanopy_inst` fields:
- Lines 1960–1965: `float(pref_forcing[p])`, `float(ztop_canopy[p])`, `float(lai_canopy[p])`,
  `float(sai_canopy[p])`, `int(ntop_canopy[p])`, `int(ncan_canopy[p])`
- Lines 1973–1974: `float(tair_profile[p, _ntop])`, `float(eair_profile[p, _ntop])`
- Lines 1989, 2002: `float(beta_canopy[p])`, `float(ustar_canopy[p])`
- Lines 2003–2013: `np.asarray(zw_profile[p])` + Python for-loop
- `_GetObu` (called from `_HF2008`): 12 × `float(mlcanopy_inst.field[p])` (lines 1205–1216)
- `_RoughnessLength` (non-diff, lines 1508–1509): `int(ntop_canopy[p])`, `int(ncan_canopy[p])`
- `_WindProfile` (non-diff, lines 1658–1659): same
- `_AerodynamicConductance` (non-diff): similar

A JAX-traceable `_HF2008_diff` **already exists** and does not have any of these issues.
It uses `_GetObu(diff_mode=True)`, `_RoughnessLength_jax`, `_WindProfile_jax`,
`_AerodynamicConductance_jax`.

#### 2. `MLSolarRadiationMod.SolarRadiation`
- Lines 154–156, 426–428, 731–733: `np.asarray(mlcanopy_inst.ncan_canopy)`,
  `np.asarray(ntop_canopy)`, `np.asarray(nbot_canopy)` — three separate function
  sections each do this to materialize the index arrays before a per-layer loop.

#### 3. `MLLongwaveRadiationMod.LongwaveRadiation`
- Lines 317–318: `int(mlcanopy_inst.ntop_canopy[p])`, `int(mlcanopy_inst.nbot_canopy[p])`

#### 4. `MLFluxProfileSolutionMod`
- Lines 87, 569, 629, 742, 848: `int(ncan[p])`, `int(mlcanopy_inst.ncan_canopy[p])`,
  `int(mlcanopy_inst.ntop_canopy[p])`

#### 5. `MLLeafPhotosynthesisMod`
- Lines 1589, 1735: `int(mlcanopy_inst.ncan_canopy[p])`

#### 6. `MLRungeKuttaMod.RungeKuttaUpdate`
- Line 106: `int(mlcanopy_inst.ncan_canopy[p])`

### Why the blockers exist

The `ncan_canopy`, `ntop_canopy`, `nbot_canopy` fields are JAX arrays because they're part
of the `mlcanopy_type` NamedTuple.  They're constant for a fixed site but stored as JAX
arrays because they were initialized that way.  When these are extracted with `int()` or
`np.asarray()` inside a JIT-traced function, JAX raises `ConcretizationTypeError`.

The `float()` calls in `_HF2008` are for a different reason: they feed into `_ObuFuncPure`
(Python `math.*` version) which is deliberately kept as Python scalars for ~50× iteration
speedup.  However, `_obu_fixed_iter` (already using `lax.fori_loop`) and `_ObuFuncPure_jax`
handle the same physics purely in JAX.

### Why NOT attempted

Fixing all blockers would require:
1. Threading `ncan_vals`, `ntop_vals`, `nbot_vals` (pre-materialized Python int tuples) as
   additional parameters through `SolarRadiation`, `LongwaveRadiation`, `FluxProfileSolution`,
   `LeafPhotosynthesis`, `RungeKuttaUpdate` — changing signatures of 6+ public functions.
2. Routing `CanopyTurbulence` through `_HF2008_diff` (or a new multi-patch JAX wrapper)
   instead of `_HF2008` in non-diff mode — changing turbulence performance characteristics
   (the Python-scalar path in `_HF2008` is faster in eager mode because it avoids JAX dispatch
   overhead per iteration).
3. Validating that the JAX-path physics results match the Python-scalar path numerically
   (they should, but require a full simulation test to confirm).

This is a significant refactoring (6+ modules, ~30 call sites) that violates the
"minimal fix" constraint and risks introducing subtle numerical differences.

### Recommended path forward (if pursued in a future session)

**Option A (low risk):** Pre-materialize `ncan_vals`, `ntop_vals`, `nbot_vals` in
`MLCanopyFluxes` (like `ncan_vals` is already done), then thread them as `static` arguments
to each physics function that needs them.  This fixes ~20 blockers.  The remaining
blockers in `_HF2008` require step 2 above.

**Option B (higher gain):** Create a new `_step_fn_jit(mlcanopy_inst, calday_ml, nstep_ml)`
that calls the diff-mode (`_HF2008_diff`-style) JAX paths for ALL physics, looping over
patches using concrete `filter_mlcan` and pre-materialized `ncan_vals`/`ntop_vals`/`nbot_vals`.
This is essentially using the diff-mode physics code paths without `jax.grad` — same XLA
program, no gradient tape.  Compile once, reuse for all 30 sub-steps.

**Option C (current status quo):** Keep the Python for-loop with eager dispatch.  Existing
`@jax.jit` wrappers on `_copy_bef_state` and `_implicit_fps_jit` already compile the
hottest inner kernels.  The remaining dispatch overhead is bounded by the 30-step loop.

---

## 2026-04-03 — NaN gradient fix: Monin-Obukhov ψ stability functions (session 5)

**Status:** Fix applied; `test_grad.py` validation run was started but did not complete within the session (context limit hit mid-run). Forward loss confirmed finite (-2549.4493). Gradient result pending.

### Root cause identified and fixed

**`_psim_monin_obukhov` and `_psic_monin_obukhov`** (`MLCanopyTurbulenceMod.py`, lines 186, 220):

The unstable branch computes `x = (1 - 16*zeta)^(1/4)`.  
For stable conditions (`zeta >= 1/16`): `1 - 16*zeta <= 0`, the stable branch is selected by `jnp.where`, but JAX still evaluates both branches' VJPs.  
`jnp.abs(1 - 16*zeta)^(1/4)` at `1 - 16*zeta = 0` → gradient = `(1/4) * 0^(-3/4) * sign(0) = inf * 0 = NaN`.  
This NaN propagates through `jnp.where`'s VJP to all downstream gradients (turbulence, leaf temperature, air profile, soil temperature).

**Fix:** Replace `jnp.sqrt(jnp.sqrt(jnp.abs(1.0 - 16.0*zeta)))` with `jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0*zeta, 1.0e-10)))`.  
`jnp.maximum` gradient is 0 when argument is below threshold (not inf), so unselected-branch VJP = `0 * finite = 0`, not NaN.  
Forward values are unchanged for any `zeta < 0` (unstable, actually selected branch).

Applied to both `_psim_monin_obukhov` (ψ for momentum) and `_psic_monin_obukhov` (ψ for scalars).

### Pending validation
- `test_grad.py` was running (1 sub-step, Euler, eager grad) but output not yet captured.
- Next session: check gradient fields `tair_profile`, `eair_profile`, `tleaf_leaf`, `tg_soil` for NaN.
- If all finite: commit fix + run full 30-step test, then begin optimization loop.

---

## 2026-04-02 — jax.checkpoint at outer loop boundary (session 4)

**Problem:** `jax.grad` OOMs (103 GB, 100+ min) on the full 6 sub-steps × 5 RK stages = 30-iteration
Python for loop. JAX unrolls all iterations at trace time; backward pass must materialize all
intermediate activations from all 30 iterations = hundreds of thousands of tensors.

**Fix:** Extract loop body into `_make_physics_step(calday, nstep)` factory returning a pure
`mlcanopy_inst → mlcanopy_inst` function. In diff mode, wrap each call with `jax.checkpoint`:
```python
mlcanopy_inst = jax.checkpoint(_step_fn)(mlcanopy_inst)
```
`jax.checkpoint` (remat) does NOT save internal activations during forward; recomputes them on
demand during backward. Memory drops from O(num_ml_steps × step_mem) to O(step_mem).

**test_grad.py change:** Also reduce to 1 sub-step + Euler (nrk=0) for NaN testing, bringing
memory from ~103 GB down to ~3–4 GB. This is sufficient to validate that NaN gradients are gone.

**Non-diff path:** Unchanged — `_make_physics_step` called without checkpoint.

---

## 2026-04-02 — Additional NaN gradient fixes (session 4)

### Root causes fixed

**4. `SolarRadiation` — `cos_zen` in `kb_ic` denominator** (`MLSolarRadiationMod.py`):
`kb_ic = jnp.minimum(gd / cos_zen, kb_max)`. At solar zenith = 90° (`cos_zen = 0`), True branch
`gd / cos_zen = inf` is still differentiated by JAX even though False branch is selected. Gradient
of True branch w.r.t. `cos_zen` = `-gd / cos_zen^2 = inf`. Then `0 * inf = NaN` in backward.
`solar_zen_forcing` is a field of `mlcanopy_inst` (the differentiated variable), so this is in
the gradient path.
Fix: `cos_zen_safe = jnp.maximum(cos_zen, 1e-10)` before division.

---

## 2026-04-02 — Additional NaN gradient fixes (session 3)

### Root causes fixed

**1. `_obu_writeback_jax` — jnp.where denominator pattern** (`MLCanopyTurbulenceMod.py`):
```python
# Before:
_dm2 = jnp.where(jnp.abs(zlog + psim) > eps, zlog + psim, eps)
ustar_val = uref_p * vkc / _dm2
```
When `zlog + psim = 0`: cond=False, `_dm2 = eps` (forward OK). But JAX differentiates the True branch `vkc / (zlog+psim)`, giving `inf`. Then `0 * inf = NaN` in backward.
Fix: `sign * max(|x|, eps)` pattern (no select op in denominator).

**2. `_AerodynamicConductance_jax` — jnp.where denominator pattern × 3** (`MLCanopyTurbulenceMod.py`):
Lines 1810, 1820, 1829: same `jnp.where(|x|>eps, x, eps)` denominator pattern.
Fix: `sign * max(|x|, eps)` pattern. Affects above-canopy conductance `gac` for 3 height intervals.

**3. `SoilResistance` — frozen-layer division by zero** (`MLPlantHydraulicsMod.py`):
`soilr1_v = log(root_dist/rr) / (2π * rld * dz * hk_v)`. When `hk_v = 0` (frozen layer), backward grad w.r.t. `rld_v` (from mlcanopy_inst) = `inf`.
Fix: `hk_v_safe = jnp.maximum(hk_v, 1e-30)`.

---

## 2026-04-02 — Eliminate D↔H syncs in RungeKuttaUpdate (MLRungeKuttaMod)

**Status:** Unified diff/non-diff code paths in `RungeKuttaUpdate`.

Removed `if _diff_mode: jax_path; else: numpy_path` split:
- **Pre-extraction**: replaced ~18 `np.asarray()` / `float()` calls per patch per RK step with direct JAX slices
- **Writeback**: removed `jnp.array(_result)` wrappers; `.at[].set()` now receives JAX arrays directly

Net: ~18 D→H + 12 H→D syncs eliminated per patch per RK step. With 4th-order RK (nrk=4) that's ~120 fewer syncs per ML sub-step in non-diff mode.

---

## 2026-04-02 — Fix NaN gradients in tridiagonal solvers (MLMathToolsMod)

**Status:** `jax.grad` completed but returned NaN for all fields. Root cause identified and fixed.

### Root causes (3 fixes)

**1. Monin-Obukhov φ functions** (`MLCanopyTurbulenceMod.py`):
`_phim` and `_phic` computed `sqrt(1-16*zeta)` without a positive guard. For stable conditions (`zeta > 1/16`), the stable branch is selected but JAX still differentiates the unstable expression, giving `sqrt(negative) = NaN`. Fixed with `jnp.maximum(1.0 - 16.0*zeta, 1e-10)`.

**2. `tridiag_2eq`** (coupled T/q solver in `_implicit_fps_jit`) divides by `det = ainv*dinv - binv*cinv`. For the top-layer boundary, `c1[n] = c2[n] = 0`, so the numerators `dinv * c1[i]` etc. are zero — but JAX differentiates `x / det` as `-x/det²` regardless, producing `0 * inf = NaN` when `det` approaches zero from numerical accumulation.

Same issue in `tridiag` (single-equation solver used in solar radiation two-stream): the running determinant `bet = b[j] - a[j]*gam[j]` could be near-zero at some layer.

### Fix (`MLMathToolsMod.py`)
- `tridiag`: replace bare `/ bet` with `/ jnp.maximum(|bet|, 1e-30)` — sign-safe, gradient-safe
- `tridiag_2eq`: replace bare `/ det` with `/ jnp.maximum(det, 1e-30)` — det > 0 for all physical M-matrices

Forward-pass results unchanged (physical systems have `bet`, `det` >> 1e-30).

---

## 2026-04-02 — Unify all LeafPhotosynthesis paths to pure JAX; eliminate D↔H syncs

**Status:** Completely eliminated numpy pre-extraction/writeback in both first and second loops of `LeafPhotosynthesis`. Eliminated turbulence redundant re-extraction. Net result: ~60+ fewer D→H+H→D syncs per ML sub-step.

### Completed

**1. MLLeafPhotosynthesisMod.py — first loop unified (11 D→H + 11 H→D eliminated)**

- Removed `if _diff_mode:` / `else:` pre-extraction split — JAX arrays used always
- Removed `_diff_mode and` guard from gs_type 0/1 and gs_type 2 vmap paths
- Deleted 253 lines of dead numpy accumulator code: duplicate gs_type 0/1 block, gs_type 2 Python scalar loop, and batch writeback
- All stomatal models (Medlyn/Ball-Berry/WUE) now use `jax.vmap` in both diff and non-diff modes

**2. MLLeafPhotosynthesisMod.py — second loop unified (16 D→H + 11 H→D eliminated)**

- Removed the `else:` numpy pre-extraction blocks (16–19 `np.asarray()` calls)
- Removed `if _diff_mode:` guard and `continue` 
- Removed the entire non-diff scalar second loop (used `_CiFuncGsPure` per layer)
- Both diff and non-diff paths now use the JAX per-layer loop directly

**3. MLCanopyTurbulenceMod.py — redundant OBU re-extraction (12 D→H eliminated)**

Non-diff `_GetObu` path was calling `_ObuFunc` after convergence, re-extracting 12 scalars. Fixed by routing through `_obu_writeback_jax` which reuses `_kwargs`.

**4. Validation figures generated** (`diags/figures/`)

- `validation_flux.png`: JAX vs Fortran RMSE: Rn=0.14, H=0.74, LE=0.94 W/m²
- `validation_profiles.png`: vertical canopy profiles at noon
- `validation_scatter.png`: scatter for all 17 flux variables

**5. Differentiability test suite** (`tests/test_differentiability.py`)

5 `@pytest.mark.slow` tests: forward pass finite, grad completes, grad finite, grad nonzero, FD check.

---

## 2026-04-01 — GPU performance optimizations and differentiability fixes

**Status:** Eliminated major device-sync bottlenecks; resolved XLA `select_divide_fusion` compilation failure blocking `jax.grad`.

---

### Completed

**1. Eliminated `np.asarray()` D↔H syncs in sun/shade leaf merge (MLCanopyFluxesMod.py:620–665)**

Replaced numpy-based sun/shade temperature/LWP merge with pure JAX `jnp.where`. Every CLM timestep previously triggered ~4 host-device round-trips per active patch; now zero.

Before:
```python
_dpai = np.asarray(mlcanopy_inst.dpai_profile[p])  # D→H sync
_tleaf_p = np.asarray(tleaf[p])                     # D→H sync
...
tleaf = tleaf.at[p, _sl, :].set(jnp.array(_tleaf_new[_sl, :]))  # H→D
```
After: all-JAX `jnp.where`/`.at[]` ops, no host copies.

**2. Pre-materialized patch hierarchy arrays (MLCanopyFluxesMod.py, _GetCLMVar, SolarRadiation, LeafPhotosynthesis)**

Calls to `int(patch.column[p])`, `int(patch.gridcell[p])`, `int(patch.itype[p])`, `float(grc.latdeg[g])` etc. forced device-to-host syncs on every timestep. Fixed by calling `np.asarray(patch.column)` once before the loop and indexing into that.

**3. Fixed XLA `select_divide_fusion` compilation failure (`jax.grad` crash)**

`jax.grad(forward_fn)` was failing with:
```
JaxRuntimeError: INTERNAL: Failed to materialize symbols: { (<xla_jit_dylib_31>, { select_divide_fusion.1 }) }
```

Root cause: XLA's CPU `select_divide_fusion` optimization pass tried to fuse `jnp.where(cond, x, fallback) / y` patterns into a single kernel, but the LLVM backend failed to compile the fused kernel.

Fix: Replaced **all** `jnp.where(x > 0, x, fallback)` used as denominators with `jnp.maximum(x, eps)` (for non-negative quantities) or `sign * maximum(|x|, eps)` (for signed quantities). The `maximum` op is NOT a `select` op at the HLO level, so XLA never triggers the fusion.

**Modules fixed** (all in the `jax.grad`/`MLCanopyFluxes` diff-mode path):

| Module | Pattern fixed | Lines |
|---|---|---|
| `MLSolarRadiationMod.py` | `dpai_safe`, `kb_ic_safe`, `p1/p2_safe`, `fs/fsha/suminc_safe` | 194, 249–263, 574–587, 847–876 |
| `MLCanopyNitrogenProfileMod.py` | `dpai_safe`, `denom_safe`, `fs/fsha_safe` | 203, 220, 225–226 |
| `MLFluxProfileSolutionMod.py` | `gs_gbv_denom` (×2), `den_l`, `den_lf` | 213, 228–231, 408–418 |
| `MLLeafFluxesMod.py` | `gleaf_denom`, `tleaf_denom` | 123, 136 |
| `MLLeafPhotosynthesisMod.py` | `gs_safe`, `gbc_safe`, secant `denom` | 291–292, 818 |
| `MLPlantHydraulicsMod.py` | `totevap_safe`, `lsc_safe` | 238, 268 |
| `MLCanopyWaterMod.py` | `total_safe`, `dpai_safe` | 189, 215 |
| `MLCanopyTurbulenceMod.py` | `obu_cur_safe` (×2), `tvstar_safe`, `aa_s`, `denom_L` (×3), `denom_Z` (×3), `_denom_m/c`, secant `denom` | 971–985, 1034–1035, 1075–1082, 1129, 255–256, 306–307, 315–316, 336–337, 345–346 |
| `MLMathToolsMod.py` | `a_safe`, `q_safe` in `quadratic()` | 542–550 |

**4. SolarRadiation now receives `grid` in diff-mode**

Previously `SolarRadiation` was called without `grid=grid` in the `_diff_mode` path, causing it to call `jnp.any()` and `endrun()` checks that are not JAX-traceable. Fixed in `MLCanopyFluxesMod.py:505`.

---

### Failed approaches

- **Tried outer `jax.jit` on `jax.grad`** — hit OOM on CPU and the `select_divide_fusion` error. Removed the outer JIT; eager grad works after fixing the select_divide patterns.
- **Tried `jax.lax.stop_gradient` on denominators** — would fix the compilation but break gradient flow through the denominator, affecting optimization. Not used.

---

### Known limitations / next steps

1. **Main ML sub-step loop is still a Python `for` loop** (MLCanopyFluxesMod.py:477). Converting to `lax.fori_loop` would eliminate ~48×15 = 720 separate GPU dispatches per CLM timestep. Blocked by: (a) `masterproc` print side effects, (b) some modules still have Python-level int() calls on JAX arrays. This is the #1 remaining bottleneck.

2. **`jax.grad` not yet tested to completion** — the test (`test_grad.py`) was confirmed to reach the grad computation; a full passing run is the next validation step.

3. **Remaining small D→H sources** (all bounded, low priority):
   - `MLFluxProfileSolutionMod.py:632-638, 746-747` — 9 syncs in error-check functions (`ErrorCheck01/02`)
   - `MLRungeKuttaMod.py:139-143` — 5 syncs per RK step for `bef`-state pre-extraction
   - `MLCanopyTurbulenceMod.py:1507,1661,1985` — 3 syncs in height-lookup functions

4. **MLSoilTemperatureMod.py** has similar `select_divide_fusion` patterns but is NOT in the `jax.grad` path (called from `clm_driver.py`, not `MLCanopyFluxes`). Leave for later.

---

### Performance notes (baseline, pre-optimization)

Profiling was run with `JAX_PLATFORMS=cpu`. GPU baseline not yet measured. Key numbers to capture after GPU testing:
- Wall time per MLCanopyFluxes call
- JIT compile time (first call)
- Memory footprint of mlcanopy_inst arrays
