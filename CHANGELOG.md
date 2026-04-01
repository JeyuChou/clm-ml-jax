# Changelog

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

3. **numpy flux accumulators** (`np.zeros` in `_MLTimeStepFluxIntegration`) force per-sub-step host transfers in non-diff mode. Would benefit from converting to JAX arrays.

4. **MLSoilTemperatureMod.py** has similar `select_divide_fusion` patterns but is NOT in the `jax.grad` path (called from `clm_driver.py`, not `MLCanopyFluxes`). Leave for later.

---

### Performance notes (baseline, pre-optimization)

Profiling was run with `JAX_PLATFORMS=cpu`. GPU baseline not yet measured. Key numbers to capture after GPU testing:
- Wall time per MLCanopyFluxes call
- JIT compile time (first call)
- Memory footprint of mlcanopy_inst arrays
