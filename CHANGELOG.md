# Changelog

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
