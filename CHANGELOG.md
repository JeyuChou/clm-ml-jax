# Changelog

## 2026-04-10 — Multisite vmap N=1,2 results (session 28)

### Multisite benchmark (job 7342743, A100)

| N | vmap_1st_s | vmap_ss_s | seq_ss_s | speedup | ms/site/step |
|---|---|---|---|---|---|
| 1 | 1175.6 | 0.722 | 0.662 | 0.92× | 722.0 ms |
| 2 | 1484.3 | 1.025 | 1.312 | **1.28×** | 512.3 ms |

Crossover at N=2. N=4 compile starting. Speedup expected to grow with N.

---

## 2026-04-10 — fd_grad_check GPU results in progress (session 28)

### alpha_sw JAX gradient on A40 GPU (job 7344785)

```
dGPP/d(alpha_sw) [JAX] = 1.070136e+01   compile: 780s
```

Finite, non-zero. Compile 780s (faster than A100 1232s — no lax.scan overhead in fd_grad_check).
alpha_tref JAX grad now compiling. Revised timeline: all 5 done ~23:40 (within 2h limit).

---

## 2026-04-10 — RK4 lax.scan benchmark confirmed on A100 (session 28)

### RK4 benchmark results (job 7344537, A100)

```
RK4  diff (lax.scan):  37.1 ms/step   compile: 0.294s (cached)
RK4  non-diff:      30390.6 ms/step   →  818× speedup
  diff_mode  loss = -2903.8322  (matches non-diff ✓)
  diff loss finite: True
RK4 gradient check: compiling now (~103 min, done ~00:26)
```

**Summary of lax.scan speedups confirmed on A100 GPU (job 7344537):**
| Mode | diff (lax.scan) | non-diff | Speedup |
|---|---|---|---|
| Euler (1 sub-step) | 38.1 ms | 2941 ms | **77×** |
| RK4 (6 sub-steps × 4 stages) | 37.1 ms | 30391 ms | **818×** |

---

## 2026-04-10 — Laxscan Euler gradient PASS on A100 GPU (session 28)

### lax.scan Euler gradient verified on A100 (job 7344537)

```
dGPP/d(alpha_tref) = -4.8692e+01   finite=True   Status: OK
grad() compile time: 1202s
```

Matches CPU result (-48.69, rel err 1.66e-9). **GPU Euler lax.scan gradient confirmed.**

RK4 section now running:
- RK4 diff (lax.scan): 0.294s compile (cached!), 37.1 ms/step
- RK4 non-diff running, then RK4 gradient compile (~103 min estimated)

---

## 2026-04-10 — fd_grad_check time limit fix + backup job (session 28)

### run_fd_grad_check.sh: extended time limit to 4h

- Original 2h limit was dangerously close to estimated runtime (~115 min for 5 GPU grad compiles)
- Extended to `--time=04:00:00` in `bashscripts/run_fd_grad_check.sh`
- Submitted backup job **7345152** with `--dependency=afternotok:7344785` (4h, any GPU)
  — only runs if job 7344785 (current, 2h limit) times out

| Job ID | What | Status |
|---|---|---|
| **7344785** | fd_grad_check (A40 g188, 2h limit) | RUNNING — baseline Euler JIT compiling |
| **7345152** | fd_grad_check backup (4h, any GPU) | PENDING (afternotok:7344785) |
| **7344537** | laxscan benchmark (4h, A100 g194) | RUNNING — Euler grad compiling |
| **7342743** | multisite vmap benchmark | RUNNING — N=1 done (0.92×), N=2 compiling |
| **7344539** | plot_benchmarks | PENDING (Dependency) |

---

## 2026-04-10 — Job management + CLI fixes (session 27, continued)

### optimize_params.py CLI dispatch fix

- `--joint` flag was unreachable: `if __name__ == "__main__"` called `main()` directly
- Fixed: routes through `_parse_and_dispatch()` which checks `--joint`
- `main()` now accepts pre-parsed `args` to avoid double-`argparse` conflict
- `run_optimize_params.sh` updated to add Phase 2 joint optimization run

### laxscan benchmark resubmitted (4-hour limit)

RK4 gradient check XLA compilation exceeded 2-hour job limit:
- Euler scan body: 1 physics step → compile 1232s ✓ (PASS: dGPP/d(alpha_tref) = -48.69)
- RK4 scan body: 5 unrolled physics steps → estimated 5× longer compile (~103 min)
- Job 7342742 cancelled at 53 min to avoid timeout; resubmitted as **7344537** (4h limit)
- plot-benchmarks resubmitted as **7344539** (depends on 7344537 + 7342743)

| Job ID | What | Status |
|---|---|---|
| **7344785** | fd_grad_check (5 params, A40 GPU g188, any-GPU) | RUNNING |
| 7344537 | laxscan benchmark (4h, A100 g194) | RUNNING — Euler grad compiling |
| 7342743 | multisite vmap benchmark (A100 g191) | RUNNING — N=1 measuring |
| 7344539 | plot_benchmarks | PENDING (Dependency) |
| 7343825 | fd_grad_check (A100 constraint) | CANCELLED (superseded by 7344785) |

7344785 submitted without `--constraint=a100` to get faster scheduling; running on NVIDIA A40 (46GB).

### Euler laxscan results (job 7344537, consistent with 7342742)
```
Euler  diff (lax.scan):  38.1 ms/step  compile: 251.6s
Euler  non-diff:       2941.4 ms/step  →  77× speedup
  diff_mode  loss = -2903.8322  (matches non-diff ✓)
  diff loss finite: True
Euler gradient check: compiling now (~20 min, same as 7342742)
```
Note: JAX compile cache not hitting for Euler forward (251s vs expected ~0s).
RK4 forward was cached (0.293s in 7342742). Euler and RK4 are different cache keys.

---

## 2026-04-10 — iota_SPA gradient verified on CPU (session 27)

**CPU test: PASS**
```
FD  dGPP/d(alpha_iota) = -2.135857e+00
JAX dGPP/d(alpha_iota) = -2.135854e+00
Relative error = 1.045e-06  →  PASS
```

iota[7] = 375.0 (CHATS7 PFT 7 default).  
Gradient path: `alpha_iota → _iota_jnp = jnp.asarray(MLpftcon.iota_SPA) → _iota_pft[pft] →`
`iota_rt (vmap in_axes=None) → _StomataEfficiencyJax → _bisect_gs_ift (IFT) → gs_opt → GPP`.

All 4 active parameters now CPU-verified:
| Parameter | CPU result | Status |
|---|---|---|
| alpha_sw    | rel err 3.68e-7 | PASS |
| alpha_tref  | rel err 1.66e-9 | PASS |
| alpha_iota  | rel err 1.05e-6 | PASS ← new |
| alpha_vcmax | rel err 1.80e-8 | PASS |
| alpha_g1_MED | N/A | INACT (gs_type=2) |

GPU confirmation via job 7343825 (still pending Resources).

---

## 2026-04-10 — Joint vcmaxpft + iota_SPA optimization implemented (session 27)

### `diags/optimize_params.py` — new joint optimization section (~200 lines added)

Key additions:
- `_run_joint(log_params)`: runs model with both `vcmaxpft_jax` (JIT-bypass) and `_set_pftcon` (iota_SPA)
- `forward_joint_gpp`, `forward_joint_le`: scalar loss functions over 2D `log_params`
- `generate_synthetic_obs_joint(vcmax_true, iota_true)`: creates synthetic GPP+LE obs for testing
- `make_joint_loss_fn(gpp_obs, le_obs, w_gpp, w_le, lam_reg)`: weighted MSE + L2 reg
- `run_joint_optimization(loss_fn, n_steps, ...)`: Adam optimizer with cosine LR schedule
- `main_joint(args)`: saves results as JSON with loss curve + parameter trajectories
- `_parse_and_dispatch()`: extended CLI with `--joint` flag and `--iota-true` arg

Usage: `python diags/optimize_params.py --joint --vcmax-true 1.2 --iota-true 0.9`

Parameter coupling: vcmaxpft controls GPP amplitude (via vcmax25top → Rubisco capacity);
iota_SPA controls WUE stomatal conductance (via _bisect_gs_ift IFT). Joint optimization
recovers both from synthetic obs.

### GPU job 7343825 still pending (5-param gradient check)

All code verified on CPU. GPU confirmation pending.

---

## 2026-04-10 — Critical bug fix: CanopyNitrogenProfile @jax.jit caches vcmaxpft as constant (session 27)

### Root cause discovered (deeper than session 26 fix)

`CanopyNitrogenProfile` is decorated with `@partial(jax.jit, static_argnums=(0, 1))`.  
At first trace, `MLpftcon.vcmaxpft` is captured as an **XLA compile-time constant** — not
a runtime JAX variable.  Subsequent calls (including FD ±eps) use the CACHED compiled
XLA computation with the original values baked in.  Module-global mutation via
`_set_pftcon` cannot override the JIT cache.

**Evidence:** FD test gave `f(1+eps) == f(1-eps) == 30.918758` for all eps — function
was literally constant in alpha_vcmax.  This was NOT just a "zero gradient" — the
output was numerically identical because the XLA function ignored the module-global update.

### Fix: add `vcmaxpft_jax=None` as explicit JAX argument

**`MLCanopyNitrogenProfileMod.py`:**
- Added `vcmaxpft_jax=None` as arg 4 (non-static, part of dynamic pytree)
- When provided: `_vcmaxpft = vcmaxpft_jax` (JAX-traced)
- When None: `_vcmaxpft = MLpftcon.vcmaxpft` (module global — default/non-diff path)
- `vcmax25top = _vcmaxpft[pft]` — uses explicit arg or module global

**`MLCanopyFluxesMod.py`:**
- Added `vcmaxpft_jax=None` to `MLCanopyFluxes` signature
- Passed through to `CanopyNitrogenProfile(num_mlcan, filter_mlcan, inst, vcmaxpft_jax)`

**`diags/fd_grad_check.py`:**
- `forward_gpp_vcmaxpft` now passes `vcmaxpft_jax=alpha * _orig.vcmaxpft` directly
- No `_set_pftcon`/`_restore_pftcon` needed

**`diags/param_sensitivity.py`:**
- `_run_inst` now accepts `vcmaxpft_jax=None`
- `_gpp_vcmax`/`_le_vcmax` pass `vcmaxpft_jax=alpha * _orig.vcmaxpft`

**`diags/optimize_params.py`:**
- `_run_with_vcmax_scale` passes `vcmaxpft_jax=alpha_vcmax * _orig.vcmaxpft`

All existing callers (driver, benchmarks) use `vcmaxpft_jax=None` (default) — no behavior change.

### Why pytree arg avoids the JIT cache hit

`vcmaxpft_jax=None` → pytree: `(mlcanopy_inst, None)` → one cache entry (constant inside)  
`vcmaxpft_jax=<jax_array>` → pytree: `(mlcanopy_inst, ShapedArray)` → NEW cache entry → fresh trace
Fresh trace means `_vcmaxpft` is a JAX-traced array, `vcmax25top = _vcmaxpft[pft]` is a
dynamic gather, and gradient flows through the full nitrogen profile path.

### Gradient path confirmed

`alpha → vcmaxpft_jax → _vcmaxpft[pft] → vcmax25top → kn(vcmax25top) → nscale → vcmax25_leaf → LeafPhotosynthesis kernel → ac → agross → GPP`

### CPU test PASSED (task byv5r8guh)

```
f(1+eps)=30.920172  f(1-eps)=30.917343
FD  dGPP/d(alpha_vcmax) = 1.414393e+01   (nonzero ✓)
JAX dGPP/d(alpha_vcmax) = 1.414393e+01
Relative error = 1.802e-08  →  PASS
```

Gradient path confirmed end-to-end on CPU.

### GPU job resubmitted

| Job ID | What | Status |
|---|---|---|
| 7343825 | fd_grad_check (5 params: sw, tref, g1, iota, vcmax) | PENDING (new fixed code) |

Cancelled job 7343434 (submitted with broken vcmaxpft injection code).

---

## 2026-04-10 — Critical bug fix: MLpftcon injection pattern (session 26, part 2)

### Root cause discovered

**ALL stomatal/vcmax parameter gradient checks were silently broken.**

The injection pattern `MLpftcon.field = alpha * original.field` fails because:
1. `MLpftcon_type` is a `typing.NamedTuple` (immutable) — `AttributeError: can't set attribute`
2. `MLpftconMod.MLpftcon = original._replace(field=alpha * original.field)` replaces the
   MODULE-LEVEL variable, but each physics module that did `from MLpftconMod import MLpftcon`
   at import time holds a DIRECT REFERENCE to the original instance — the module-level
   variable replacement does NOT propagate to those local bindings.

Verified with explicit test: after `_pmod.MLpftcon = new_inst`, `_NitroMod.MLpftcon` still
points to the original. So `vcmax25top = MLpftcon.vcmaxpft[pft]` in CanopyNitrogenProfile
returned the original value regardless of alpha → gradient was 0 (constant function).

### Fix: update ALL physics module local references

Correct pattern (implemented in all three diags):
```python
def _set_pftcon(new_inst):
    MLpftconMod.MLpftcon          = new_inst   # update module variable
    MLLeafPhotosynthesisMod.MLpftcon  = new_inst   # update local binding (g1, iota, gsmin)
    MLCanopyNitrogenProfileMod.MLpftcon = new_inst  # update local binding (vcmaxpft, clump)

def _restore_pftcon():
    # restore all three back to _orig_pftcon
```

### Files fixed (commit 1aad306)

- `diags/fd_grad_check.py`: `_set_pftcon`/`_restore_pftcon` helpers; all 3 forward functions
- `diags/param_sensitivity.py`: same; also upgraded iota_SPA from FD-only to JAX grad
- `diags/optimize_params.py`: `_run_with_vcmax_scale` now uses `_set_pftcon`

### Jobs resubmitted

| Job ID | What | Status |
|---|---|---|
| 7343434 | fd_grad_check (5 params: sw, tref, g1, iota, vcmax) | PENDING |

Cancelled 7343159 (had broken injection pattern).

---

## 2026-04-10 — Gradient audit + vcmaxpft added to grad check (session 26)

### Differentiability status audit

**Verified PASS (from prior sessions):**
- `dGPP/d(alpha_sw)`:   rel error 3.68e-7  — PASS (job 7315181)
- `dGPP/d(alpha_tref)`: rel error 1.66e-9  — PASS (CPU; IFT fix for Obukhov)

**Code-fixed, GPU verification submitted (job 7343159):**
- `dGPP/d(alpha_g1_MED)`:  fixed session 25; expected INACT (gs_type=2 default)
- `dGPP/d(alpha_iota_SPA)`: fixed session 25 (IFT via _bisect_gs_ift); expected PASS
- `dGPP/d(alpha_vcmaxpft)`: newly added this session; expected PASS

### vcmaxpft gradient path confirmed differentiable

`MLpftcon.vcmaxpft` is a JAX `jnp.full` array. The gradient path is:
```
alpha * MLpftcon.vcmaxpft[pft]   (module-attr mutation at trace time)
→ vcmax25top  (MLCanopyNitrogenProfileMod.py:147)
→ vcmax25_leaf (set in mlcanopy_inst by CanopyNitrogenProfile, called at MLCanopyFluxesMod.py:577)
→ _vcmax25_p (JAX slice, passed to vmapped photo kernel)
→ vcmax_leaf (T-response)  → ac (Rubisco)  → agross → GPP
```
Also: `kn = jnp.exp(0.00963 * vcmax25top - 2.43)` when kn_val < 0 — differentiable (jnp.exp).

### Code changes

- `diags/fd_grad_check.py` (commit aba7fdd): added `forward_gpp_vcmaxpft(alpha)` — 5th
  gradient check parameter. Saved `_orig_vcmaxpft = MLpftcon.vcmaxpft`; mutates module attr
  at trace time and restores after. Reports as PASS/FAIL/INACT in summary table.

### Jobs submitted

| Job ID | What | Status |
|---|---|---|
| **7343159** | fd_grad_check (5 params: sw, tref, g1, iota, vcmax) | PENDING (Priority) |

### Architecture audit: float() calls are NOT on gradient paths

Audited all `float(mlcanopy_inst.*)` and `float(MLpftcon.*)` calls in physics modules:
- `MLCanopyTurbulenceMod.py` lines 831-842: in `_ObuFunc` callback (secant iteration).
  Used only via `_obu_fixed_iter` whose output is `stop_gradient`'d in diff mode.
  Gradient flows through `_ObuFuncPure_jax` (JAX-traced) via IFT, not through `_ObuFunc`.
- `MLLeafPhotosynthesisMod.py` lines 465-491: in `_CiSolverCallbackPure` (old CI solver).
  Not called in diff mode — vmapped kernels handle all computation (gs_type 0/1/2 all use vmap).
- All other `float()` calls are in diagnostic/non-diff paths, driver code, or setup code.

**No remaining float() calls on active gradient paths.**

---

## 2026-04-10 — Differentiability fix: g0, g1_MED, iota_SPA, gsmin_SPA (session 25)

### Problem

`d(GPP)/d(g1_MED)` and `d(GPP)/d(iota_SPA)` were structurally broken.
The stomatal conductance parameters (`g0_MED`, `g1_MED`, `g0_BB`, `g1_BB`,
`iota_SPA`, `gsmin_SPA`) were converted to Python floats via `float(_np[pft])`
before being closed over in the lru_cache'd kernel factory functions. This broke
the autodiff tape — JAX saw them as constants, not differentiable parameters.

### Fix (`src/multilayer_canopy/MLLeafPhotosynthesisMod.py`, commit `0305f82`)

Moved all stomatal parameters from closed-over factory arguments to **runtime
JAX broadcast scalars** (`in_axes=None` in vmap), following the existing pattern
used for `vcmaxse_rt`/`vcmaxc_rt` in the acclim kernels.

- `_make_leaf_photo_kernel`: removed `g0_val`/`g1_val` from `lru_cache` key; added
  `g0_rt`/`g1_rt` as runtime kernel args. vmap `in_axes` updated to `(0,)*11 + (None, None)`.
- Same for `_make_leaf_photo_kernel_acclim` + `_get_vmapped_photo_kernel_acclim`.
- `_make_leaf_photo_kernel_wue`: removed `iota_pft`/`gsmin_pft` from cache key;
  added `iota_rt`/`gsmin_rt` as runtime args. vmap `in_axes` updated to include 2 more Nones.
- Same for `_make_leaf_photo_kernel_wue_acclim` + `_get_vmapped_photo_kernel_wue_acclim`.
- `LeafPhotosynthesis` call sites: `_g1_MED_jnp = jnp.asarray(MLpftcon.g1_MED)` instead
  of `np.asarray` + `float()`. JAX scalars passed as broadcast args to vmapped kernels.

### Gradient paths enabled

- `d(GPP)/d(g1_MED)`: `alpha_tref → T_leaf → LeafPhotosynthesis → _ci_solver_scan → g1_rt → gs → ci → GPP`
- `d(GPP)/d(iota_SPA)`: via IFT in `_bisect_gs_ift → _StomataEfficiencyJax(iota=iota_rt) → GPP`

### Status

Code committed and pushed. No benchmark jobs needed to verify — will add a
`d(GPP)/d(g1_MED)` and `d(GPP)/d(iota_SPA)` check to `diags/fd_grad_check.py`.

### Previously broken (not yet verified)

- `d(GPP)/d(g1_MED)` — now should return finite non-zero value
- `d(GPP)/d(iota_SPA)` — now should return finite non-zero value via IFT

---

## 2026-04-10 — Benchmark jobs submitted + plotting infrastructure (session 24)

### Jobs submitted to measure post-optimization performance

| Job ID | Script | What it measures | Wall |
|---|---|---|---|
| **7342742** | `run_laxscan_benchmark.sh` | diff (lax.scan) vs non-diff (Python loop), Euler + RK4 — ms/step, compile time, grad check | 2h |
| **7342743** | `run_multisite_benchmark.sh` | GPU vs CPU, N=1,2,4,8,16,32, full RK4 — ms/site/step, vmap speedup | 6h |
| **7342744** | `run_plot_benchmarks.sh` | Generates `benchmark_summary.png` — runs only after 7342742 + 7342743 succeed | 10m |

Jobs 7342742/7342743 PENDING (Resources/Priority) as of 2026-04-10.
Job 7342744 has `--dependency=afterok:7342742:7342743`.

**Previous baseline** (jobs 7329052 + 7329441, pre-optimization):
- Euler diff: 37.2 ms/step  |  non-diff: 6,095 ms  |  speedup: 164×
- RK4   diff: 37.8 ms/step  |  non-diff: 117,164 ms |  speedup: 3101×
- Compile time: ~290s (Euler), ~2h+ (RK4)

**Expected "after"**: same steady-state ms (kernel unchanged), lower compile time
from persistent cache + smaller XLA trace from vectorization; non-diff notably
faster from reduced dispatch count.

### New benchmark infrastructure

- `diags/benchmark_laxscan.py`: now saves `diags/figures/laxscan_benchmark.csv`
  (diff_ss_ms, nondiff_ss_ms, speedup, compile_s for Euler + RK4)
- `diags/benchmark_multisite.py` (`run_multisite_benchmark.sh`): updated to run
  `--backend both` with N=1..32 for GPU and CPU (was GPU N=1..32 + CPU N=1..8)
- `diags/plot_benchmarks.py`: 3-panel comprehensive figure
  - Panel A: lax.scan speedup bar chart (Euler + RK4, diff vs non-diff)
  - Panel B: ms/site/step vs N (log–log), GPU + CPU, Fortran reference line
  - Panel C: GPU vs CPU throughput at N=1, 16, 32 (sites/second, speedup labels)
  - Output: `diags/figures/benchmark_summary.png` (300 dpi)
- `bashscripts/run_plot_benchmarks.sh`: CPU-only SLURM job for plot generation

### Results to check when jobs complete
- `logs/7342742_laxscan_benchmark.out` — post-opt timing + grad check status
- `logs/7342743_multisite_benchmark.out` — GPU/CPU N-site scaling table
- `diags/figures/laxscan_benchmark.csv` + `multisite_benchmark.csv`
- `diags/figures/benchmark_summary.png` — comprehensive plot

---

## 2026-04-10 — Gradient explosion fix: Obukhov-length IFT (session 24)

### Root cause of `dGPP/d(alpha_tref) = 9.95e+144`

The gradient explosion came from differentiating through `_obu_fixed_iter`,
a 25-iteration secant solver for the Obukhov stability length.

**Gradient path:** `alpha_tref → thref/thvref → _GetObu (secant, 25 iters) → obu →
_obu_writeback_jax → ustar → wind_profile → LeafBoundaryLayer → gbc/gbv →
LeafPhotosynthesis → GPP`

Differentiating through 25 secant iterations accumulates Jacobians: `|J|^25 ≈ 1e144`.
The stomatal bisection (`_bisect_gs_ift`) was already fixed with IFT in session 21.
The Obukhov solver was the remaining source of explosion.

### Fix: IFT applied to `_GetObu` (diff_mode=True)

Same Newton-refinement pattern as `_bisect_gs_ift`:

```python
obu0    = stop_gradient(_obu_fixed_iter(..., n_iter=25))
f0      = _ObuFuncPure_jax(obu0, **kwargs)            # gradient flows through theta
df_dobu = stop_gradient(FD of F w.r.t. obu)           # denominator frozen
obu_ift = obu0 - safe_f0 / safe_denom                 # IFT Newton step
```

- Forward: `F(obu0) ≈ 0` → `obu_ift ≈ obu0 = obu*` ✓
- Backward: `d(obu_ift)/dtheta = -∂F/∂theta / stop_grad(dF/dobu)` = IFT ✓

**File:** `src/multilayer_canopy/MLCanopyTurbulenceMod.py` — `_GetObu` diff_mode branch.

### Verification (CPU, isolated unit test)

```
obu* = 8.0909 m   F(obu*) = 0.00e+00   obu_ift = 8.0909 m (forward correct)
d(obu)/d(thvref):  JAX=5.40e-02  FD=5.38e-02  rel=3.88e-03  PASS
d(obu)/d(thref):   JAX=-7.53e+00  FD=-7.50e+00  rel=3.88e-03  PASS
d(obu)/d(rhomol):  JAX=0.00e+00   FD=0.00e+00   rel=0.00e+00  PASS
```

### Why `z0m` (RoughnessLength_jax) does NOT need IFT

`z0m_canopy` is only used in non-diff-mode diagnostics (`_CanopyFluxesDiagnostics`,
`_MLScaleAndWriteBack`). It is NOT on the gradient path from alpha_tref to GPP.

### Next step

Submit `run_fd_grad_check.sh` on A100 to confirm `dGPP/d(alpha_tref) ≈ -49`
(matching FD value from job 7315181, rel error < 1%).

---

## 2026-04-10 — GPU performance optimizations (session 24)

### Changes implemented (all committed to `differentiable-physics`)

**1. JAX Persistent Compilation Cache** — eliminates 290s recompile on subsequent runs
- Added `JAX_COMPILATION_CACHE_DIR` + `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10` to
  all SLURM scripts: `run_laxscan_benchmark.sh`, `run_multisite_benchmark.sh`,
  `run_fd_grad_check.sh`, `run_optimize_params.sh`, `run_param_sensitivity.sh`
- Added programmatic cache config to `src/offline_executable/main.py` (uses same env var if set)
- Cache dir: `/burg-archive/home/al4385/.cache/jax_compile_cache`

**2. Vectorized LAI/SAI/PAI layer loop** (`MLCanopyFluxesMod.py`)
- Replaced `for ic in range(1, _ncan+1): dlai.at[p,ic].set(...)` (3×ncan=138 scalar scatter ops)
  with 3 slice scatter ops: `dlai.at[p, _sl].set(frac * lai_val)`
- Reduces XLA trace depth inside lax.scan; improves kernel fusion

**3. JIT-compiled forcing-save helper** (`MLCanopyFluxesMod.py`)
- Added `_save_bef_forcing(filter_mlcan, inst)` JIT function (fuses 8 scatter ops → 1 XLA dispatch)
- Replaced the verbose post-loop forcing update with a single `_save_bef_forcing(...)` call

**4. Batched sun+shade LeafBoundaryLayer** (`MLLeafBoundaryLayerMod.py`)
- Added `_gb_layers_both`: outer vmap over sun/shade dim of the existing `_gb_layers` layer-vmap
- Added `LeafBoundaryLayerBoth`: processes isun and isha in one GPU kernel dispatch
- Updated `MLCanopyFluxesMod.py` call site: 2 sequential `LeafBoundaryLayer` calls →
  1 `LeafBoundaryLayerBoth` call (halves boundary-layer dispatch count per RK stage)

**5. Vectorized second loop in LeafPhotosynthesis** (`MLLeafPhotosynthesisMod.py`)
- Replaced `for ic in range(1, _ncan_p+1)` scalar loop (46×~10 ops) with slice operations
- All water-stress + photosynthesis-recompute ops are now element-wise over `_sl = slice(1, n+1)`
- `is_c3` and `gspot_type` are Python statics → resolved at trace time (no `jnp.where` overhead)
- Reduces second-loop XLA ops from ~460 scalar to ~10 slice ops per `il`

**6. AOT compilation script** (`diags/aot_compile.py`, `bashscripts/run_aot_compile.sh`)
- Strategy 1: JIT warm-up to populate persistent cache; reports whether cache hit on 2nd call
- Strategy 2 (--export flag): `jax.export` serialization to `.mlirbc` artifact (JAX ≥ 0.4.25)
- Run with `sbatch bashscripts/run_aot_compile.sh` to pre-warm the cache on A100

### Expected impact
| Change | Impact |
|---|---|
| Persistent cache | Eliminate 290s compile on every run after first |
| LAI/SAI vectorization | ~138 fewer scalar XLA ops per sub-step |
| Forcing save JIT | 8 → 1 Python→GPU dispatch per CLM timestep |
| LeafBoundaryLayer both | 2 → 1 GPU dispatch per RK stage |
| LeafPhotosynthesis 2nd loop | ~460 → ~10 XLA scalar ops per il per sub-step |

### Next steps
- Run `sbatch bashscripts/run_aot_compile.sh` to populate the persistent cache
- Re-run `benchmark_laxscan.py` to measure improvement from vectorization changes
- Confirm full-physics vmap results from job 7328915 for paper Section 4

---

## 2026-04-09 — lax.scan benchmark results (jobs 7329052 + 7329441, A100 GPU)

### Benchmark: lax.scan diff mode vs Python loop non-diff mode

**Euler (1 sub-step, 0 RK stages):**

| Mode | Steady-state time | Speedup |
|---|---|---|
| Diff mode — lax.scan | 37.2 ms | — |
| Non-diff — Python for-loop | 6,095 ms | **164×** |

**Full RK4 (6 sub-steps × 4 RK stages):**

| Mode | Steady-state time | Speedup |
|---|---|---|
| Diff mode — lax.scan | 37.8 ms | — |
| Non-diff — Python for-loop | 117,164 ms (117s!) | **3101×** |

**Correctness:**
- Euler: diff mode loss = −2903.8322; non-diff sum(shair+etair) ≈ −2903.8322 ✓
- RK4: diff mode loss = −2903.8322; non-diff shair sum = −3041.1166 (expected: different because non-diff averages over sub-steps)
- Both diff losses finite ✓

**Gradient check (Euler):**
- `dGPP/d(alpha_tref) = 9.95e+144` — finite but pathologically large (gradient explosion)
- Indicates the physics AD path has gradient instability; needs investigation (likely
  unstable branch through stomatal/photosynthesis solvers under RK4 composition)
- RK4 gradient check compilation hit 2h SLURM wall limit before completing

**Why the 3101× speedup for RK4?**
- Non-diff RK4: 6 sub-steps × 4 RK stages = 24 separate Python→GPU dispatch round-trips
- Diff mode: XLA fuses everything into one kernel; compile once, run in 38ms flat.

### Why 207× speedup?
- Non-diff mode: Python loop issues **one GPU dispatch per sub-step** (and in Euler,
  even 1 sub-step costs 6.3s because each call re-triggers Python→XLA overhead for
  every `jnp.` operation in the physics chain).
- Diff mode (lax.scan): XLA sees the entire loop as **one fused kernel**; compiles
  once (290s), then runs in 30.6 ms.
- lax.scan also enables `jax.checkpoint` for O(step_mem) backward memory.

### Files
- `diags/benchmark_laxscan.py` — benchmark script
- `bashscripts/run_laxscan_benchmark.sh` — SLURM job script

---

## 2026-04-09 — Session 23: gradient check jobs, optimize_params submission

### Fixes committed
- `diags/benchmark_laxscan.py` (commit 9054134): fixed gradient check kwargs filter.
  Now only filters `atm2lnd_inst` and `wateratm2lndbulk_inst` (keeps `grid`+`_o2ref_py`).
  Previous version removed `grid` and `_o2ref_py` causing MLCanopyFluxes to fail.

### Jobs running
- 7329012: fd_grad_check — running (verifying bracket_ok fix, dGPP/d(alpha_sw) + dGPP/d(alpha_tref))
- 7329322: param_sensitivity — running (vcmax25 JAX grad finiteness check)
- 7329437: opt-params — pending (200-step Adam optimization, vcmaxpft identifiability test)
- 7328915: multisite-vmap full RK4 — running 90+ min (full RK4 JIT compile is very slow)

### SLURM fix
- `bashscripts/run_optimize_params.sh`: reduced `--time` from 8h to 2h, removed `--qos=hpc_test`
  (hpc_test has 2h wall limit), added `--partition=short` for longer jobs.

---

## 2026-04-09 — Optimization experiment implementation (session 22, continued)

### Files created (all committed to `differentiable-physics`)

**`diags/param_sensitivity.py`** — prerequisite sanity check for optimization.
Computes dGPP/dθ and dLE/dθ at CHATS7 operating point for:
- `alpha_sw`, `alpha_tref`: JAX autodiff (existing paths)
- `alpha_vcmax25`: JAX autodiff via module-global mutation of `MLpftconMod.MLpftcon`
  — differentiable because `CanopyNitrogenProfileMod.py` reads `MLpftcon.vcmaxpft[pft]`
  via JAX dynamic gather (no `np.asarray` → `float()` in critical path)
- `alpha_iota`: FD only — `iota_SPA` is extracted as `float(_iota_np[pft])` inside
  the kernel factory in `MLLeafPhotosynthesisMod.py` (lines 2067+). Making it
  JAX-differentiable requires passing `iota` as a runtime arg to vmapped kernels
  (Phase 2 refactoring — documented in code, not yet done).

**`diags/expt_load_obs.py`** — AmeriFlux observation loader. Reads FLUXNET2015 FULLSET
CSV, applies daytime/u*/QC filters, aligns to model timesteps.

**`diags/optimize_params.py`** — Phase 1: vcmaxpft recovery from synthetic CHATS7 obs.
Adam optimizer with cosine LR annealing. Synthetic case: recover vcmaxpft=125 from
CLM default 57.7. Gradient flows via module-global mutation (same as param_sensitivity).

**`src/multilayer_canopy/MLpftconMod.py`** — added `make_pft_params(theta_dict)` factory
function for clean parameter injection. Supports `(field_name, pft_idx)` tuple keys.

### Jobs pending
- 7329012: fd_grad_check (bracket_ok fix verification) — pending (Resources)
- 7328915: multisite benchmark full RK4 — running (1h+, stuck in JIT compile)
- 7329052: laxscan benchmark — running, still in warmup
- 7329181: param_sensitivity — pending (Priority)

---

## 2026-04-09 — lax.scan benchmark results (job 7329052)

### Euler diff mode (1 sub-step, 0 RK stages) — CONFIRMED

| Mode | First call | Steady-state |
|---|---|---|
| DIFF MODE (lax.scan) | 290s (JIT compile) | **30.6 ms/step** |
| NON-DIFF MODE (Python loop) | 6.6s | 6338.8 ms/step |
| **Speedup** | — | **207×** |

- Loss finite: True. Correctness: diff=-2903.83, non-diff sh+et=-2904.07+0.24=-2903.83 ✓
- Gradient check: FAILED with `grad requires float inputs` — needs float-only wrapper.
  Root cause: `jax.grad(forward_fn)(mlcanopy_inst)` fails because `mlcanopy_inst` has int fields.
  FIX NEEDED: wrap `jax.grad` to differentiate w.r.t. a scalar alpha (like `fd_grad_check.py`).
- Full RK4 benchmark: NOT run (job exited after grad check error).

**Key insight:** 207× speedup is from lax.scan + JIT dispatch reduction.
The Python loop (non-diff mode) re-dispatches to GPU 1 time per step (Euler).
lax.scan compiles everything into a single XLA kernel, eliminating Python overhead.

---

## 2026-04-09 — lax.scan over RK sub-steps (session 22)

### lax.scan refactoring — COMPLETE (commit 68de426)

**Goal:** Replace Python for-loop over ML sub-steps with `jax.lax.scan` in diff
mode so XLA sees one dispatch instead of `num_ml_steps` separate traces, enabling
better fusion and O(step_mem) backward memory via `jax.checkpoint`.

**Changes in `src/multilayer_canopy/MLCanopyFluxesMod.py`:**
- `nstep_ml = 0` initialised before `_physics_step_fn` (closure fix for diff mode)
- DIFF MODE: builds `_calday_arr` (pre-computed calday per sub-step), runs
  `jax.lax.scan(_scan_body_fn, mlcanopy_inst, _calday_arr)` with optional
  `jax.checkpoint` (disabled by `CLM_ML_NO_CHECKPOINT=1`)
- NON-DIFF MODE: Python loop calls new `_MLAccumulateFluxes` per step then
  `_MLScaleAndWriteBack` once after the loop
- `_MLAccumulateFluxes` and `_MLScaleAndWriteBack` added (split from old function)
- `_MLTimeStepFluxIntegration` removed (replaced by two-function split)

**Why split matters:** `_MLAccumulateFluxes` has no Python conditionals on
`nstep_ml`, and `ncan_vals` is pre-computed (no D→H syncs). This is a prerequisite
for future `lax.fori_loop` usage once all non-diff mode physics are XLA-traceable.

**Diff mode works because:**
- All physics in diff mode (grid ≠ None) are JAX-traceable
- `TimeInterpolation3` already uses `jnp.where` for traced `calday_ml`
- `CanopyTurbulence` in diff mode calls `_HF2008_diff` which ignores `nstep_ml`

**Non-diff mode unchanged in correctness:** accumulators zero-initialised by
`jnp.zeros` before loop; no reinit needed inside loop.

---

## 2026-04-09 — Fortran baseline timing + full-physics vmap benchmark (session 20)

### Fortran CLM-ml-v2 timing (CONFIRMED, Exp 5b)
- **Platform:** NCAR Derecho, Intel Ice Lake CPU, gfortran 12
- **Run:** CHATS7 May 2007, 31 days, 1488 half-hourly timesteps
- **Wall-clock time: 1m 52.6s (112.6s)**
- Compilation fixes required for gfortran portability:
  - `MLLeafPhotosynthesisMod.F90`: removed duplicate `private :: RealizedRate` (ACCESS spec error)
  - `Makefile`: added `-ffree-line-length-none` to gfortran build line (lines >132 chars)
  - Both fixes pushed to `AyaLahlou/CLM-ml_v2.CHATS` main branch

### Performance comparison (partial — full-physics vmap pending)
| Scenario | Platform | Wall time |
|---|---|---|
| Fortran single-site, 31 days | Derecho CPU | **112.6s** |
| JAX single-site, per-step (steady) | A100 GPU | ~100s/step |
| JAX single-site, 31 days (extrapolated) | A100 GPU | ~41 hrs |
| JAX vmap N=32, per-step (Euler physics) | A100 GPU | 0.394s/step |
| JAX vmap N=32, 31 days (Euler, extrapolated) | A100 GPU | ~586s |

- Euler benchmark (job 7315861) used reduced physics (1 sub-step, 0 RK stages)
- Full-physics benchmark (job 7328915, A100) submitted with `--full-physics` flag
  (runge_kutta_type=41, dtime_ml=300s, 6 sub-steps × 4 RK stages)
- Full-physics results needed to confirm vmap speedup numbers for paper

### Paper update (JAXES.tex, Section 4 + Limitations)
- Hardware line: updated to Derecho/Intel Ice Lake for Fortran timing
- Limitations paragraph: updated with exact Fortran wall time (112.6s),
  ~1300× JAX single-site slowdown, and preliminary 3.5× vmap N=32 result

---

## 2026-04-09 — Multi-site vmap benchmark (session 18)

**Status:** Benchmark script and SLURM job created. Job 7315861 pending on A100.

### fd_grad_check results (job 7315181, A100): MIXED
- dGPP/d(alpha_sw):   JAX=1.070e+01, FD=1.070e+01, rel=3.68e-07 **PASS** ✓
- dGPP/d(alpha_tref): JAX=6.575e+144, FD=-4.869e+01, rel=1.35e+143 **FAIL** — gradient explosion

**alpha_tref explosion TRUE root cause (found session 21):**
Not the `1e-15` denominator. Root cause is `bracket_ok=False` layers.
When the WUE bisection has no root (dark layers, bracket_ok=False), `gs_opt=gsmin`
but `f(gsmin) ≠ 0` (can be O(0.01)). The Newton step:
`gs_ift = gsmin - f(gsmin)/df_dgs = 0.002 - (-0.016)/0.16 = 0.104`
...extrapolates far from gsmin. This wrong `gs_ift` has a huge gradient.

**False fix (session 19, job 7328527):** `|df/dgs| > 1e-6` guard.
df_dgs is 0.16 (well-conditioned) for dark layers, so guard doesn't trigger.
Explosion persists despite this fix.

**Real fix (session 21, job 7329012):** Gate Newton correction on `bracket_ok`.
`_bisect_gs_jax` now returns `(gs_opt, bracket_ok)`.
`apply = bracket_ok & (|df/dgs| > 1e-6)`.
When `apply=False`: `gs_ift = gs0 = gsmin` (no Newton step, zero gradient).
CPU test: `d(agross)/dT` passes at rel=1.66e-9. Submitting GPU verification.
File: `src/multilayer_canopy/MLLeafPhotosynthesisMod.py` lines 1541-1572.

### Plan
Root cause of JAX slowdown vs Fortran: single-column model = ~0% GPU occupancy.
46-element arrays cannot saturate thousands of CUDA cores.
Fix: run N sites simultaneously via `jax.vmap` — GPU time per site drops ~Nx.

### Implementation: thin wrapper at driver level only
No physics files changed. `MLCanopyFluxes` in `_diff_mode` (grid=GridInfo) is
already pure-functional. vmap operates on it from outside.

**Files created:**
- `diags/benchmark_multisite.py`: benchmark script
  - Initializes 1 CHATS7 site via `expt_init.py`
  - Stacks N copies of `mlcanopy_inst` via `jax.tree_util.tree_map(stack, ...)`
  - Times `jit(vmap(_single_site_step))(batched)` vs N sequential `jit(step)` calls
  - Sweeps N = 1, 2, 4, 8, 16, 32
  - Reports: vmap_first_s, vmap_steady_s, seq_total_s, speedup, ms/site/step
  - Saves `diags/figures/multisite_benchmark.csv`
- `bashscripts/run_multisite_benchmark.sh`: SLURM script (A100, 4 hr, 64GB)

**Key insight:** `vmap` over leading batch dim of `mlcanopy_inst` works because:
- `_diff_mode` uses `grid.ncan` (Python int) instead of `int(mlcanopy_inst.ncan_canopy[p])`
- All `for p in filter_mlcan:` loops use `p=1` (concrete Python int)
- `.at[1].set()` broadcasts correctly under vmap over leading N dim

### Expected result
- Sequential N=1: baseline ~Xs/step
- vmap N=8: ~8x throughput (near-linear scaling if compute-bound)
- vmap N=32: saturate A100 → near-constant wall time → ~32x sites/second
- GPU vmap vs CPU seq: demonstrates GPU value for ensemble runs

---

## 2026-04-09 — IFT fix for WUE bisection gradient (session 17)

**Status:** Newton-refinement IFT implemented. Jobs 7314582/7314583 pending on A100 to verify.

### Root cause confirmed (job 7314440)
- Stage 1 d(apar)/d(alpha_sw): JAX=2408.725, FD=2408.725, rel=1.18e-09 **PASS** — solar rad gradient EXACT
- Stage 2 d(agross)/d(alpha_sw): JAX=18.20, FD=21.69, rel=16.1% **FAIL** — bug in photosynthesis
- Stage 3 dGPP/d(alpha_sw): JAX=9.008, FD=10.693, rel=15.76% **FAIL**

### Fix: `_bisect_gs_ift` — IFT-based gradient for WUE stomatal bisection

**File:** `src/multilayer_canopy/MLLeafPhotosynthesisMod.py`

The WUE stomatal optimization (`gs_type=2`) uses `_bisect_gs_jax` (20-iteration bisection
via `jax.lax.fori_loop` + `jnp.where` branch selection). Differentiating through the bisection
gives wrong gradients because `jnp.where` propagates gradients through BOTH branches.

**Initial attempt**: `jax.lax.custom_root` → FAILED due to JAX version issue:
`tangent_solve` received a `Partial` (function) instead of a scalar,
causing `TypeError: unsupported operand type(s) for /: 'Partial' and 'DynamicJaxprTracer'`.

**Final fix**: Newton-refinement identity avoids `custom_root` entirely:
```python
gs_ift = gs0 - f(gs0; theta) / stop_grad(df/dgs)
```
where `gs0 = stop_gradient(bisect result)`.
- Forward: `f(gs0) ≈ 0` → `gs_ift ≈ gs0 = gs*` (correct root value)
- Backward: `d(gs_ift)/d(theta) = -∂f/∂theta / stop_grad(df/dgs) = IFT` ✓
- `df/dgs` computed via finite differences with `stop_gradient` (denominator only)
- `f(gs0; theta)` computed once with JAX array gradients flowing (provides IFT numerator)

Both WUE kernels (`_make_leaf_photo_kernel_wue` and `_make_leaf_photo_kernel_wue_acclim`)
updated to use `_bisect_gs_ift`.

### SLURM node constraint fix
- Added `--constraint=a100` to all grad-check SLURM scripts
- V100S nodes (--constraint=v100s) have broken conda env (base anaconda + ~/.local = conflict)
- rtx8000 nodes (Quadro RTX 8000) have same issue
- A100 nodes (g189-g193, g283-g284) work correctly

### New diagnostic scripts
- `diags/test_bisect_ift.py`: targeted test of `_bisect_gs_ift` gradient (single layer, ~5-10min)
- `run_test_bisect_ift.sh`: SLURM job script for above

### Verification: CONFIRMED (job 7314583, A100 node)

- Stage 1 d(apar)/d(alpha_sw):  rel=1.18e-09 **PASS** (unchanged)
- Stage 2 d(agross)/d(alpha_sw): rel=**1.76e-07 PASS** (was 16.1% — fix confirmed)
- Stage 3 dGPP/d(alpha_sw):      rel=**3.68e-07 PASS** (was 15.76% — fix confirmed)

Newton-refinement IFT works. The ~15% gradient error for alpha_sw is fully explained
by the WUE bisection gradient bug (not jnp.where kinks as hypothesized in session 16).

**Job 7315181 (fd_grad_check, full run, A100 g284): IN PROGRESS**
Testing both alpha_sw and alpha_tref gradients. Expected to complete soon.

---

## 2026-04-08 — Gradient discrepancy root cause identified: non-smooth physics (session 16)

**Status:** Stage 3 of isolate_grad_path (job 7314377) confirmed SAME discrepancy as fd_grad_check.
JAX=9.008, FD=10.693, rel_err=15.76%. jax.checkpoint RULED OUT (CLM_ML_NO_CHECKPOINT=1 used).
Root cause: non-smooth photosynthesis operations. CHANGELOG updated.

### Root cause of 15% gradient discrepancy: non-smooth operations in Ci solver

isolate_grad_path Stage 3 (no checkpoint): JAX=9.008e+00, FD=1.069e+01, rel=15.76% FAIL.
Matches fd_grad_check exactly — this is NOT a checkpoint artifact.

The CHATS7 canopy has `fracsun≈0.09` (only ~9% sunlit) and `ncan=46` layers.
Most canopy mass is shaded, with some lower layers having near-zero net photosynthesis.

Non-smooth operations that create zero-subgradient kinks:
1. `jnp.where(anet_val < 0.0, jnp.zeros(()), ci_dif)` in `_CiFuncPure_jax` (line 771)
   — When anet < 0, secant step is truncated. JAX subgradient = 0 at threshold.
   FD perturbs alpha_sw upward and those layers may cross to anet > 0.
2. `jnp.maximum(_agross, 0.0)` — clamps gross photosynthesis to non-negative.
3. `jnp.maximum(ci_val - cp_val, 0.0)` — Rubisco-limited rate clamping.

The 76% discrepancy for alpha_tair is larger because temperature change affects
Vcmax/Jmax via nonlinear enzyme kinetics, hitting more threshold crossings.

**Paper framing**: Acknowledge this limitation honestly. The gradients are FINITE and
PARTIALLY CORRECT (accurate for active layers far from threshold). The ~15% error
means optimization may require more iterations but converges (as shown by Exp 4).
Smooth alternatives (softplus for max ops) would fix this but change physics.

### Baseline canopy state (CHATS7, midday 2007-05-01)
```
fracsun_profile[p, 1:5] = [0., 0.0937, 0.0958, 0.0991]  (upper canopy very shaded)
dpai_profile[p,   1:5] = [0., 0.0179, 0.0312, 0.0457]
apar_leaf[p, 1:5, isun] = [0., 625., 634., 640.] µmol/m²/s  (high, light-saturated)
agross_leaf[p, 1:5, isun] = [0., 12.55, 12.62, 12.74] µmol/m²/s
GPP (baseline) = 30.92 µmol/m²/s
```

Sunlit leaves have high PAR and are RUBISCO-limited (Ac < Aj). Shade leaf PAR is much
lower; some deep layers approach the anet=0 threshold → discontinuous gradient boundary.

---

## 2026-04-08 — Isolation bug identified: spval cancellation + Fortran fix resubmitted (session 16)

**Status:** FD=0 in Stage 1/2 of isolate_grad_path is a float64 cancellation bug, NOT a gradient bug.
JAX timing at 32/48 timesteps (~101s/step steady-state on RTX 8000). Fortran job resubmitted with fix.

### Root cause of Stage 1/2 FD = 0 (isolate_grad_path.py)

`apar_leaf` and `agross_leaf` have `spval=1e36` for inactive canopy layers.
Plain `jnp.sum(inst.apar_leaf[p, 1:ncan+1, :])` gives a baseline of ~2e38 (dominated by spval).
When alpha is perturbed by eps=1e-4, the signal change (~1e4) is ~34 orders of magnitude below
the float64 noise floor at ~2e38 (machine epsilon * 2e38 ≈ 4e22). So a_plus - a_minus = 0 exactly.

**Fix**: use dpai-weighted sum: `jnp.sum(dpai[:, None] * apar)`. Since dpai=0 for inactive layers,
`dpai * spval = 0`, eliminating spval contamination. FD now measures only active layers.

Applied fix to `diags/isolate_grad_path.py` (Stages 1 and 2). Resubmitted as job 7314440.

### photo_kernel_grad.py (job 7314396) FAILED

`ImportError: cannot import name 'pftconMod' from 'multilayer_canopy'` — old version of script.
Current `diags/test_photo_kernel_grad.py` is fixed (imports from `clm_src_main.pftconMod`).
Pending resubmission as job 7314421 (via `run_test_photo_kernel_grad.sh`).

### Fortran timing (job 7314403) FAILED — second compilation error

Error: `Arithmetic overflow converting REAL(8) to INTEGER(4) at (1). Use '-fno-range-check'`
Fix: added `-fno-range-check` to GFORTRAN_CMPLR in `run_fortran_timing.sh`.
Resubmitted as job 7314439.

### JAX timing (job 7314322) IN PROGRESS

32/48 timesteps complete. Steady-state: ~101-102s/timestep on Quadro RTX 8000 (46GB).
Extrapolated 31-day run: ~44 hours.

---

## 2026-04-08 — Gradient isolation: custom_vjp ruled out, Exp 5 timing submitted (session 15)

**Status:** grad_mode_comparison (job 7314294) COMPLETE. custom_vjp in tridiag_2eq is NOT the bug.
Isolation jobs running: 7314377 (stage-by-stage), 7314396 (photosynthesis kernel). Exp 5 timing
jobs: 7314322 (JAX, GPU), 7314403 (Fortran, gfortran). Stage 1 FD still pending.

### Key results from grad_mode_comparison (job 7314294)

Both flux_profile_type=1 (custom_vjp ACTIVE) and flux_profile_type=0 (custom_vjp ABSENT) give:
- alpha_sw: rel_err=1.129e-01 (11.3%)
- alpha_tair: rel_err=7.016e-01 (70.2%)

**Conclusion: custom_vjp in tridiag_2eq/ImplicitFluxProfileSolution is NOT the root cause.**

Caveat: mode=0 likely uses stale JIT cache from mode=1 (flux_profile_type is Python module var,
not a JAX static arg). But the mode=1 result alone is conclusive: bug exists even in the single
Euler step where agross_leaf is set by LeafPhotosynthesis BEFORE FluxProfileSolution runs.
Therefore tridiag_2eq is NOT in the gradient path for d(agross)/d(alpha_sw).

### Gradient path analysis

In a single Euler step (num_ml_steps=1, nrk_steps=0):
```
alpha_sw → forc_solad → swskyb_cur → swskyb_forcing
         → SolarRadiation → apar_leaf (depends linearly on alpha)
         → LeafPhotosynthesis → agross_leaf
         → FluxProfileSolution  [runs AFTER LeafPhoto, does NOT affect agross]
         → compute_gpp → GPP
```
- `fracsun_profile` does NOT depend on alpha_sw (only canopy geometry)
- `CanopyTurbulence (_HF2008_diff)` does NOT read rnleaf_leaf or depend on alpha_sw
- `tridiag_2eq` in FluxProfileSolution NOT in gradient path

### New diagnostics running

1. **isolate_grad_path.py** (job 7314377, CLM_ML_NO_CHECKPOINT=1):
   - Stage 1: d(sum(apar_leaf))/d(alpha_sw) — should = sum(apar) if solar radiation gradient correct
   - Stage 2: d(sum(agross_leaf))/d(alpha_sw) — isolates photosynthesis gradient
   - Stage 3: d(GPP)/d(alpha_sw) — should reproduce 9.008 vs 10.693 discrepancy

2. **test_photo_kernel_grad.py** (job 7314396, CLM_ML_NO_CHECKPOINT=1):
   - Calls vmapped photosynthesis kernel DIRECTLY with baseline leaf data
   - Tests d(sum(agross))/d(apar_scale) at the kernel level
   - If PASS: bug is in data flow (MLCanopyFluxes level), not in kernel
   - If FAIL: bug is inside the photosynthesis kernel

### Code inspection findings (no bug found in these)

- `_copy_bef_state`: pure `.at[].set()` ops, fully differentiable. Ruled out.
- `GetAtmForcing`: reads `tref_cur_forcing` / `swskyb_cur_forcing` which ARE set from alpha-scaled
  `atm2lnd_inst` in MLCanopyFluxes.__init__. Chain is intact. Ruled out.
- `tridiag_2eq` custom_vjp math: solves adjoint system A^T λ = g correctly (IFT).
  Called once (not in iteration loop). `defvjp` registered at module level. Ruled out.
- `jax.checkpoint` around `_physics_step_fn`: isolate_grad_path.py disables this with
  `CLM_ML_NO_CHECKPOINT=1`, so its Stage 1 result will isolate checkpoint as a cause.

### Remaining suspects
1. `float()` or `np.asarray()` cast somewhere in `SolarRadiation` or `LeafPhotosynthesis`
   disconnecting the JAX trace (most likely given identical error in both flux_profile_type modes).
2. `jax.checkpoint` interaction — ruled out if isolate_grad_path Stage 1 passes.
3. `fracsun_profile` used as weights in `compute_gpp` — if it has a `stop_gradient` or is
   computed non-differentiably in `SolarRadiation`.

### Experiment 5: Runtime comparison (jobs submitted)

JAX timing (job 7314322):
- 1-day run (48 timesteps), GPU (A100 40GB)
- First timestep: 325s (JIT compilation)
- Steady-state: ~108s/timestep
- Extrapolated 31-day run: ~44 hours (much slower than Fortran expected)

Fortran timing (job 7314403, resubmit of failed 7314346):
- First attempt (7314346) FAILED: `nvfortran: Command not found` after `module load nvhpc/25.1`
- Fix: switched to `gfortran` (`/usr/bin/gfortran` GCC 8.5.0, no module needed)
- NetCDF: `/burg/opt/netcdf-fortran-4.6.2` (confirmed on disk), NetCDF-C at `/burg/opt/netcdf-c-4.9.3`
- Makefile patched at runtime: `make "cmplr=gfortran -O2 -L... -lnetcdff -lnetcdf -lblas -lm"`
- Runs `./prgm.exe < nl.CHATS7.05.2007` (31-day, May 2007)

### JAX version: 0.9.2

---

## 2026-04-08 — Gradient diagnostic: jacfwd ruled out, flux_profile_type=0 test queued (session 14 cont.)

**Status:** job 7314173 (grad_mode_comparison) failed with TypeError. jacfwd impossible.
Next: rerun with flux_profile_type=0 vs 1 comparison (job TBD).

### jacfwd is impossible through custom_vjp (JAX limitation)

Attempted `jax.jvp(forward_gpp_sw, ...)` to bypass `custom_vjp` in `ImplicitFluxProfileSolution`
and compare forward-mode gradient vs FD. JAX raised:

```
TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function.
```

This is a hard JAX constraint — `custom_vjp` functions block all forward-mode AD.
The `jacfwd` diagnostic strategy is ruled out entirely.

### New diagnostic plan: flux_profile_type=0 vs 1

The only remaining isolation test: switch `MLclm_varctl.flux_profile_type = 0` (well-mixed
turbulence, no iterative solver, no `custom_vjp`) and compare `jax.grad` vs FD.

- If `jacrev ≈ FD` at `flux_profile_type=0` but `jacrev ≠ FD` at `flux_profile_type=1`
  → `custom_vjp` in `ImplicitFluxProfileSolution` is the confirmed bug.
- If `jacrev ≠ FD` at both modes → bug is elsewhere (NaN gradients, stop_gradient, etc.).

`grad_mode_comparison.py` rewritten to:
- Drop jacfwd (crashes)
- Test both flux_profile_type=0 and 1
- FD at 5 epsilons per mode
- Print automated conclusion

---

## 2026-04-08 — Paper experiments: job results + gradient discrepancy diagnosis (session 14)

**Status:** Jobs 7312754/55/56 completed. Gradients non-zero (3-layer fix confirmed working).
New finding: 4th zero-gradient layer (Vcmax25/dpai overwritten by physics). Gradient accuracy
test fails — suspected cause: incorrect `custom_vjp` in `ImplicitFluxProfileSolution`.

### Experiment results (jobs 7312754/55/56)

**Exp 2 — Gradient correctness (fd_grad_check, job 7312754): FAIL**
```
dGPP/d(alpha_sw)   JAX=9.008e+00   FD=1.069e+01   rel_error=15.8%   FAIL
dGPP/d(alpha_tair) JAX=-8.559e+01  FD=-4.869e+01  rel_error=75.8%   FAIL
```
- Gradients are **non-zero** — the 3-layer fix (compute_gpp via agross_leaf + atm2lnd_inst scaling) worked.
- Signs are physically correct (+SW→GPP, +Tair→-GPP at high T).
- BUT JAX and FD disagree significantly; fails <1% criterion.
- Suspected root cause: `ImplicitFluxProfileSolution` uses `custom_vjp`. If the custom adjoint
  is incorrect/approximate, `jax.grad` gives wrong values while FD (evaluating forward function
  directly) gives the right answer. Temperature shows larger discrepancy (76%) because alpha_tair
  gradient path passes through the turbulence solver more heavily than alpha_sw (15%).
- **Not yet investigated.** Three diagnostics to run:
  1. FD at multiple epsilons — confirm FD is stable (rules out nonlinearity / wrong eps)
  2. Switch `flux_profile_type=0` (no implicit solver, no custom_vjp) — if JAX≈FD, custom_vjp is culprit
  3. Compare `jacfwd` vs `jacrev` vs FD — `jacfwd` bypasses custom_vjp; if jacfwd≈FD≠jacrev, confirmed

**Exp 3 — Jacobian sensitivity (sensitivity_analysis, job 7312755): COMPLETED**
```
Raw Jacobian J[output, param]:
               Vcmax25      T_air     SW_rad   q_humidity      dpai
GPP           0.000e+00  -8.559e+01  9.008e+00  -2.701e-01   0.000e+00
H (top)       0.000e+00   3.005e+04 -2.131e+00   9.530e+01   0.000e+00
LE (top)      0.000e+00  -5.842e-02  5.464e-06  -2.729e-04   0.000e+00
```
- Non-zero gradients for T_air, SW_rad, q_humidity — working.
- Vcmax25 and dpai (LAI) are **still zero** — see 4th zero-gradient layer below.
- Timing: jacrev=1094.8s vs FD estimate ~118.1s (single fwd=11.8s × 10 evaluations).
  jacrev is ~9× SLOWER than FD because backward pass through custom_vjp is expensive.
  (This is the opposite of the claimed speedup — revise paper framing.)

**Exp 4 — Calibration demo (calibration_demo, job 7312756): COMPLETED**
```
Method              Final alpha   |error|    Final loss   N evals  Wall time
Adam (grad-based)   1.000760      7.60e-4    6.92e-8       100      1548s
Nelder-Mead         1.000000      0.000000   4.94e-16       42      465s
```
- Both methods recover alpha_sw=1.0 from 0.7 perturbation.
- Nelder-Mead converges to machine precision in 42 evals; Adam reaches only 7.6e-4 error in 100 evals.
- For this 1D problem, gradient-free Nelder-Mead outperforms Adam. The gradient-based advantage
  appears at higher dimensionality (O(p) evals for gradient vs O(2p) for FD-based gradient-free).
  Paper should clarify this caveat.

### 4th zero-gradient layer: Vcmax25 and dpai recomputed inside physics

Even after all 3 previous fixes, Vcmax25 and dpai show zero Jacobian entries. Root cause:

- **`vcmax25_profile` / `vcmax25_leaf`**: recomputed from scratch every call by `CanopyNitrogenProfile`
  (called at line 524 of `MLCanopyFluxesMod.py` inside `_physics_step_fn`). The value comes from
  `MLpftcon.vcmaxpft[pft]` — a fixed PFT lookup constant — not from the `mlcanopy_inst` field.
  Scaling `vcmax25_profile` in `modified_ml` has zero effect; it's immediately overwritten.

- **`dpai_profile`**: recomputed in `MLCanopyFluxes.__init__` (lines 388-394) from
  `dlai_frac_profile * elai_patch` where `elai_patch` comes from `canopystate_inst`. Scaling
  `dpai_profile` in `modified_ml` is overwritten before the physics runs.

To get non-zero Vcmax25 gradient: need to scale `MLpftcon.vcmaxpft` (global) or pass a scale
factor into `CanopyNitrogenProfile` as a traced argument. This requires modifying `_physics_step_fn`.
Not attempted yet.

### Script label fix needed
`sensitivity_analysis.py` docstring and print statement say "jacfwd" but code uses `jax.jacrev`.
Labels need updating (edit was not applied this session).

---

## 2026-04-08 — Paper experiments: diagnose zero-gradient root cause (session 13)

**Status:** Root cause fully traced (3 layers deep). Jobs 7312754/55/56 resubmitted. Waiting on results.

### Zero gradient diagnosis

Exp 2–4 scripts were getting gradient = 0 for all outputs. Root cause identified in two parts:

**Part A — Radiation overwrite bug:**
`MLCanopyFluxes.__init__` (lines 910-921) overwrites `swskyb_cur_forcing` and `swskyd_cur_forcing`
from `atm2lnd_inst.forc_solad_downscaled_col` and `forc_solai_grc`. Similarly, `tref_cur_forcing`
is overwritten from `atm2lnd_inst.forc_t_downscaled_col`, and `qref_cur_forcing` from
`wateratm2lndbulk_inst.forc_q_downscaled_col`. Scaling fields in `mlcanopy_inst` has NO effect
because they are immediately overwritten in the `for fp in range(...)` loop.

**Fix:** All experiment scripts now scale `atm2lnd_inst` and `wateratm2lndbulk_inst` directly
(using `._replace(forc_solad_downscaled_col=alpha*...)`) and pass them as traced arguments
to `MLCanopyFluxes`. `_mlcf_kwargs_no_atm` excludes `atm2lnd_inst` / `wateratm2lndbulk_inst`
so they can be passed as separate traced args.

**Part B — Physical light-limitation:**
`dGPP/d(alpha_vcmax25) ≈ 0` is physically correct: CHATS7 walnut orchard at noon in May
is strongly light-limited (Aj < Ac for all layers), so Vcmax25 doesn't limit GPP.

### Sensitivity analysis result (job 7312602 — COMPLETED with wrong forward fn)
All Jacobian entries near zero because of Part A above. Needs re-run with fixed forward_multi.

### Third zero-gradient layer: diff-mode skips CanopyFluxesDiagnostics

`MLCanopyFluxes` sets `_diff_mode = grid is not None` (line 242). In diff mode, `_CanopyFluxesDiagnostics`
is SKIPPED (`if not _diff_mode`, line 644). This function is the ONLY place that sets `gppveg_canopy`.
So `gppveg_canopy` is always stale (pre-step value) when called from our gradient experiments (which
always pass `grid`).

`agross_leaf` IS updated by `LeafPhotosynthesis` inside `_physics_step_fn` (which always runs).

**Fix:** Added `compute_gpp(inst, p, ncan)` helper in `expt_init.py` that reads `agross_leaf` and
computes the fracsun/dpai-weighted sum — exactly what `_CanopyFluxesDiagnostics` does but without
the Python-level flux accumulator.

### Scripts updated
- `diags/expt_init.py`: exports `atm2lnd_inst`, `wateratm2lndbulk_inst`, `isun`, `isha`, `compute_gpp`
- `diags/sensitivity_analysis.py`: forward_multi scales atm2lnd_inst; uses compute_gpp; output [GPP, H_top, LE_top]
- `diags/fd_grad_check.py`: tests dGPP/d(alpha_sw) and dGPP/d(alpha_tref) via atm2lnd_inst + compute_gpp
- `diags/calibration_demo.py`: calibrates alpha_sw via atm2lnd_inst + compute_gpp

### Oracle validation updated to 31-day run
Table 1 and all paper metric strings updated to 1488 timesteps (31 days). H RMSE=0.063, GPP exact.

---

## 2026-04-06 — Fix NaN gradients for differentiability (session 12)

**Status:** COMPLETE. All gradients finite. Validated on GPU (V100S).

### Validation result (Euler, 1 sub-step, JIT, no checkpoint)

```
grad(tair_profile ): FINITE  max_abs=7.8665e+05
grad(eair_profile ): FINITE  max_abs=9.0199e+02
grad(tleaf_leaf   ): FINITE  max_abs=3.3488e+02
grad(tg_soil      ): FINITE  max_abs=4.3466e+01
```

Forward loss = -2903.83 (finite). Grad compilation time: 861s (V100S at 75% load).

### Problem

`jax.grad` of the full CLM-ml forward pass produced NaN for `tair_profile`
and `tleaf_leaf` gradients.  `eair_profile` and `tg_soil` gradients were
finite, indicating the NaN source was in temperature-only code paths.

### Root cause

JAX evaluates **both branches** of `jnp.where(cond, f(x), g(x))` during
the backward pass.  When one branch contains `x ** n` (0 < n < 1) or
`1/x` with x = 0, the gradient is inf.  Even though the branch is masked
to zero, `0 * inf = NaN` propagates.

### Fixes applied

| Module | Pattern | Fix |
|--------|---------|-----|
| `MLLeafBoundaryLayerMod.py` | `re**0.5`, `re**0.8`, `gr**0.25` at zero wind/neutral | `re_safe = jnp.maximum(re, 1e-30)` before powers |
| `MLCanopyWaterMod.py` | `(h2ocan/h2ocanmx)**0.67` base = 0 | `fwet_base = jnp.maximum(..., 1e-30)` before power |
| `MLPlantHydraulicsMod.py` | `1/soilr_v`, `1/nlayers_f`, `1/rld_v` at zero | `jnp.maximum(denom, eps)` guards on all denominators |
| `MLMathToolsMod.py` | `sqrt(max(disc, 0))` discriminant = 0 | Floor changed from `0.0` to `1e-30` |
| `MLCanopyTurbulenceMod.py` | `sqrt(max(..., 0))` in `_GetBeta_jax` both branches | Floor changed from `0.0` to `1e-30` |

### Diagnostic scripts created

- `diags/bisect_nan_grads.py` — stop_gradient bisection to isolate NaN modules
- `diags/debug_tridiag_vjp.py` — instruments tridiag_2eq custom VJP backward
- `diags/quick_grad_check.py` — fast JIT-based gradient NaN validation

### Key finding: tridiag_2eq custom VJP is NOT the source

Instrumented backward confirmed both cotangent inputs (g_t, g_q) and
adjoint outputs (grad_d1, grad_d2) are finite.  The NaN enters in the
gradient chain from flux-profile RHS coefficients back to temperature
inputs, through the boundary layer conductance and canopy water modules.

---

## 2026-04-03 — Fix jit(scan) recompilation in _obu_fixed_iter (session 11)

**Status:** Root cause identified and fixed. Committed and pushed.

### Root cause

`_obu_fixed_iter` (in `MLCanopyTurbulenceMod.py`) used `jax.lax.fori_loop`
with a `body` lambda that **closed over `kwargs`** — a dict of Python floats
extracted from JAX arrays (one set per patch per sub-step).  Each unique
combination of float values produced a structurally different Python function
object.  Since `jax.lax.fori_loop` is implemented via `jax.lax.scan`, JAX
compiled a new XLA `scan` program for every patch on every sub-step.

With 1 patch × 30 sub-steps × ~3 Obukhov solver calls = **~90 scan
recompilations per CLM timestep**, each taking ~0.68s → ~61s overhead per
timestep.

Detected with `JAX_LOG_COMPILES=1` in `diags/debug_recompile.py`:
- Before fix: 90 `Compiling jit(scan)` events in 3 timesteps
- After fix: 1 `Compiling jit(scan)` (initial only) — 0 on subsequent timesteps

### Fix applied (`src/multilayer_canopy/MLCanopyTurbulenceMod.py`)

1. Defined `_obu_body_pure` at **module level** (not as a closure inside
   `_obu_fixed_iter`).
2. All 12 patch-level constants (`Lc_p`, `ztop_p`, ...) now pass through the
   `fori_loop` **carry tuple** as JAX scalars rather than being baked into the
   trace as Python float literals.
3. `_obu_fixed_iter` now converts all kwargs to `jnp.asarray(...)` and packs
   them into the initial carry before calling `jax.lax.fori_loop`.

### Timing results (3 CLM timesteps, CHATS7, V100S ~75% occupied)

| Timestep | Before fix | After fix |
|---|---|---|
| 1 (compile+run) | ~342s | ~313s |
| 2 (steady-state) | ~146s | ~120s |
| 3 (steady-state) | ~146s | ~120s |

~18% improvement in steady-state. The remaining ~120s is GPU execution
time (zero recompilations in timesteps 2-3), not JIT overhead.

### Analysis of remaining 120s

With 0 jit/XLA activity on timestep 2+, the 120s is pure computation:
- 30 sub-steps × 3 RK iterations × ~12 physics kernels = ~1080 kernel calls
- ~111ms per kernel call average (on a 75%-occupied V100S)
- This is compute-limited, not JIT-limited

Possible next optimizations (diminishing returns):
- Profile GPU utilization per sub-step to identify stalls
- Fuse physics kernels with `jax.vmap` across sub-steps (if memory allows)
- Reduce number of sub-steps / RK iterations at acceptable accuracy loss

---

## 2026-04-03 — GPU benchmark post lru_cache fix (session 10)

**Status:** Benchmark complete. Significant improvement measured; further recompilation still occurring.

### Results

| Metric | Before (baseline) | After (lru_cache fix) | Speedup |
|---|---|---|---|
| Compile + first run | 1987 s | 342.563 s | **5.8x faster** |
| Steady-state mean | 1942 s | 146.6 s | **13.2x faster** |
| Steady-state min | — | 145.5 s | — |
| Steady-state std | — | 1.16 s | — |

GPU: Tesla V100S-PCIE-32GB (cuda:0), JAX 0.9.2. Note: both V100S GPUs were ~75% occupied by other processes (~24.8 GB / 32 GB) during the run, which likely inflated compile and run times.

Saved to: `diags/figures/benchmark_post_lru_cache.txt`

### Analysis

The lru_cache fix eliminated the majority of the recompilation overhead (60 XLA
recompiles per CLM timestep → ~1). However, steady-state is still 146 s/step
(expected was <<1 s). This suggests further recompilation is occurring on each
timestep, likely from a different source than the leaf photo kernels.

**Possible remaining causes:**
- Other module-level closures (e.g. turbulence, longwave radiation) that still
  create new Python objects per sub-step and escape lru_cache.
- Per-timestep Python scalar arguments being passed to JIT functions as
  non-static values (changing each step → cache miss).
- The `calday_ml` float passed to `_physics_step_fn` — if this is still a
  Python float rather than a JAX traced value, each new float causes a new
  JIT trace.

**Next steps:** Profile individual sub-steps to identify which module is still
recompiling. Check `MLCanopyTurbulenceMod`, `MLLongwaveRadiationMod`, and the
main `_physics_step_fn` closure for non-cached kernel factories.

---

## 2026-04-03 — Fix XLA recompilation in LeafPhotosynthesis (session 9)

**Status:** Root cause identified and fixed. Committed. Benchmark completed (session 10).

### Root cause (clarified)

The CHANGELOG session 8 entry noted "steady-state = 1942s (also recompiling)".
The actual root cause was **not** `calday_ml` being passed to JIT functions
(it isn't — `GetAtmForcing` is not JIT-compiled, and no other JIT function
receives `calday_ml`). The true cause was:

`_make_leaf_photo_kernel` and `_make_leaf_photo_kernel_wue` returned **new
closure objects** on every call (each with different closed-over constants
that vary per-PFT per-CLM-timestep). The call sites did:

```python
kernel = _make_leaf_photo_kernel(...)    # new closure each call
vmapped = jax.vmap(kernel, in_axes=0)   # new traced fn each call → recompile
```

Called 30 sub-steps × 2 leaf types × (1 first loop) = 60 times per CLM
timestep, this triggered 60 XLA compilations — most of the 1942s wall time.

### Fix applied (`src/multilayer_canopy/MLLeafPhotosynthesisMod.py`)

1. Added `@functools.lru_cache` to `_make_leaf_photo_kernel` and
   `_make_leaf_photo_kernel_wue` — same closure returned for same PFT params.
2. Added `_get_vmapped_photo_kernel` and `_get_vmapped_photo_kernel_wue`
   (also `lru_cache`'d) that return `jax.jit(jax.vmap(kernel))`.
3. For WUE kernel: `o2ref_p` and `pref_p` promoted from closed-over values
   to explicit `in_axes=None` broadcast args — so per-sub-step pressure
   changes don't invalidate the JIT cache.
4. All PFT-level constants (`g0_val`, `g1_val`, `iota_pft`, etc.) converted
   to Python `float()` for `lru_cache` hashability.

### Verification

- `lru_cache` returns same object for same PFT args: ✓
- Second call executes in 0.000s (no recompile): ✓
- WUE kernel with different `pref_p`: 0.000s (no recompile): ✓

### What remains

- Full benchmark re-run to measure actual wall-clock improvement (GPU
  was occupied during this session).
- The 1942s baseline likely becomes ~30 × kernel_exec_time (<<1s each)
  per CLM timestep on GPU.

---

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
