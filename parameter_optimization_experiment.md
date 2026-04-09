# Parameter Optimization Experiment Design: Gradient-Based Calibration of CLM-ML-JAX Plant Traits

**Version:** 1.0  
**Date:** 2026-04-09  
**Branch:** `differentiable-physics`  
**Status:** Pre-implementation design document

---

## 1. Scientific Motivation

### 1.1 The parameter uncertainty problem in land surface models

Land surface models represent photosynthesis, transpiration, and carbon exchange through a set of plant physiological parameters whose true values are uncertain and spatially heterogeneous. The default parameter tables in CLM — `vcmaxpft` in `src/multilayer_canopy/MLpftconMod.py`, `iota_SPA`, `g1_MED`, and related entries — are single representative values per Plant Functional Type (PFT), estimated from small samples of leaf-level gas exchange measurements. Aboelyazeed et al. (2023, *Biogeosciences*) demonstrated that these default CLM4.5 values are biased for multiple PFTs and that learning them from tower flux data with a differentiable framework reduced RMSE for net photosynthesis rates from 6.92 to 4.73 μmol m⁻² s⁻¹ (a 32% improvement) while improving correlation from 0.539 to 0.744 across 43 sites.

The multilayer canopy model used in CLM-ML-JAX introduces an additional source of parameter sensitivity beyond single-layer models. Bonan et al. (2021, *Agricultural and Forest Meteorology*) showed that multilayer canopy models (with 5–15 layers) capture vertical profiles of leaf water potential, temperature, and absorbed radiation that a one-layer model cannot represent. These vertical gradients matter because stomatal conductance responds to local microclimate at each layer, not canopy-mean conditions. Consequently, parameters that regulate how light and temperature modulate stomatal behavior — `iota_SPA` (WUE efficiency), `vcmaxpft` (Rubisco capacity), and `g1_MED` (Medlyn slope) — have qualitatively different sensitivity in a multilayer context than in a big-leaf model.

The CHATS walnut orchard paper (Bonan et al. 2025/2026, *Agricultural and Forest Meteorology*) demonstrates this concretely: changing `Vcmax25` from 100 to 125 to 150 μmol m⁻² s⁻¹ produces measurable changes in modeled latent heat flux and GPP throughout the growing season (their Fig. 6), and their sensitivity analysis confirms that the model is particularly sensitive to `Vcmax25` and to the stomatal closure relationship `f(ψl)` parameterized by `psi50_gs` and `shape_gs`.

### 1.2 Why gradient-based optimization goes beyond existing methods

Traditional parameter estimation methods face a fundamental computational bottleneck: every function evaluation requires running the full model forward. Manual tuning, Latin hypercube sampling, and MCMC all require O(10³–10⁵) evaluations. For a 31-day CHATS7 simulation at 30-min resolution, each forward run currently takes ~112 seconds on a Fortran CPU reference (confirmed CHANGELOG.md session 20). Evaluating 1,000 candidate parameter sets would require ~31 CPU-hours per site.

Gradient-based optimization collapses this to O(10–100) evaluations because `jax.grad` computes the full gradient `dL/dθ` for all parameters simultaneously in a single backward pass, at roughly the cost of 3–5 forward passes (for reverse-mode AD through the 4th-order Runge-Kutta integrator). The key capability unlocked in this project is correct autodiff through the WUE bisection solver via an Implicit Function Theorem (IFT) Newton correction applied only when `bracket_ok=True` (CHANGELOG.md session 21). This fix ensures `dGPP/d(alpha_sw)` matches finite differences at 3.68 × 10⁻⁷ relative error.

The differentiable physics-informed ML (dPL) paradigm pioneered by Aboelyazeed et al. (2023) showed that treating the physics model as a differentiable black box, and backpropagating through it from a flux-observation loss, recovers parameters that are physically meaningful (same order of magnitude as trait database values) and produce better predictions than default values. CLM-ML-JAX extends this to the full multilayer canopy with 46 layers, 4th-order Runge-Kutta integration, roughness sublayer turbulence closure, and WUE stomatal optimization — a substantially more complete and theoretically grounded model than the single-layer photosynthesis module used in Aboelyazeed et al. (2023).

---

## 2. Parameter Selection

### 2.1 Primary optimization targets

The following parameters from `src/multilayer_canopy/MLpftconMod.py` are selected based on (a) demonstrated sensitivity in the literature, (b) availability of observational priors, and (c) identifiability under the available flux data.

**Tier 1 — High priority (optimize first):**

| Parameter | Symbol in code | Units | Default (PFT 7) | Literature range | Rationale |
|---|---|---|---|---|---|
| Max carboxylation rate | `vcmaxpft[pft]` | μmol m⁻² s⁻¹ | 57.7 (CHATS: 125) | 20–150 | Dominant control on GPP under Rubisco-limited conditions; shown by Aboelyazeed et al. (2023) to be most impactful parameter |
| WUE efficiency | `iota_SPA[pft]` | μmol CO₂ mol⁻¹ H₂O | 750 (CHATS: 375) | 200–1000 | Directly governs stomatal optimization in WUE mode; sets the water cost of carbon gain |
| Medlyn slope | `g1_MED[pft]` | kPa^0.5 | 4.45 (broadleaf deciduous temperate) | 1.5–8.0 | Controls stomatal response to VPD; well-constrained by eddy covariance LE |

**Tier 2 — Secondary (optimize jointly with Tier 1 if compute allows):**

| Parameter | Symbol in code | Units | Default | Literature range | Rationale |
|---|---|---|---|---|---|
| Clumping index | `clump_fac[pft]` | dimensionless | 1.0 | 0.5–1.0 | Controls effective LAI seen by radiation; Bonan et al. (2021) showed canopy integration error is sensitive to LAD profile |
| Leaf water potential at 50% conductance loss | `psi50_gs[pft]` | MPa | -2.3 (CHATS: -1.60) | -4.0 to -1.0 | Governs drought response; Bonan et al. (2025/2026) showed f(ψl) relationship strongly affects LE and H |
| Stomatal closure shape | `shape_gs[pft]` | dimensionless | 40.0 | 5–60 | Coupled to psi50_gs; determines abruptness of stomatal closure |

**Tier 3 — Structural (after Tier 1–2 converge):**

| Parameter | Symbol in code | Units | Default | Notes |
|---|---|---|---|---|
| Stem hydraulic conductance | `gplant_SPA[pft]` | mmol m⁻² s⁻¹ MPa⁻¹ | 4.0 (CHATS: 7.0) | Identifiable via LE diurnal shape |
| Min stomatal conductance | `gsmin_SPA[pft]` | mol m⁻² s⁻¹ | 0.002 | Nighttime LE constraint |

### 2.2 Parameter bounds and priors

All parameters should be transformed to log space or constrained via sigmoid transforms to enforce positivity and physical bounds:

- `vcmaxpft`: [20, 200] μmol m⁻² s⁻¹. Prior: log-normal with μ=ln(62.5), σ=0.5 (from CLM defaults).
- `iota_SPA`: [100, 1500] μmol CO₂ mol⁻¹ H₂O. Prior: log-normal with μ=ln(750), σ=0.7.
- `g1_MED`: [1.0, 12.0] kPa^0.5. Prior: log-normal with μ=ln(4.45), σ=0.4.
- `clump_fac`: [0.4, 1.0]. Prior: beta distribution on [0.4, 1.0].
- `psi50_gs`: [-4.0, -0.5] MPa. Prior: normal with μ=-2.3, σ=0.5.

The Jmax25/Vcmax25 ratio is held fixed at 1.67 × Vcmax25 (as in `MLLeafPhotosynthesisMod.py`, consistent with the Bonan et al. (2025/2026) CHATS parameterization, their Table 2).

---

## 3. Site and Data Selection

### 3.1 Site strategy

The 15 AmeriFlux sites implemented in `src/offline_driver/TowerDataMod.py` span 5 PFTs:

| PFT | PFT index | Sites |
|---|---|---|
| Broadleaf deciduous temperate tree | 7 | US-Ha1, US-MMS, US-UMB, US-Dk2, CHATS7, UMBSmw |
| Needleleaf evergreen boreal tree | 2 | US-Ho1, US-Me2 |
| Needleleaf evergreen temperate tree | 1 | US-Dk3 |
| C3 non-arctic grass | 13 | US-Var, US-Dk1 |
| C3 crop | 15 | US-IB1, US-Ne3, US-ARM, US-Bo1 |

**Recommended calibration set (Phase 1):** CHATS7 only. This site has the richest observational dataset (above-canopy fluxes, within-canopy profiles, wind tunnel data from the CHATS campaign), the best-characterized parameters (Bonan et al. 2025/2026 Table 2), and the `pftcon_val=1` override block in `MLpftconMod.py` provides independently derived starting values for PFT 7. The CHATS7 dataset covers May 2007, giving 1488 half-hourly timesteps with daytime GPP and LE well above noise.

**Recommended extension (Phase 2):** Add US-UMB (PFT 7, deciduous temperate), US-Ha1 (PFT 7), US-Me2 (PFT 2), and US-Var (PFT 13) to achieve multi-PFT coverage. This mirrors the approach of Aboelyazeed et al. (2023) who used 43 sites across 9 PFTs and demonstrated that parameters learned across PFTs on a shared loss function generalize better.

**Phase 3 (vmap):** All 15 sites simultaneously using `jax.vmap` over the site dimension, with PFT-specific parameters. This requires vectorizing `MLCanopyFluxes` over a batch leading dimension — the architecture for this is already demonstrated in `diags/benchmark_multisite.py`.

### 3.2 Calibration and validation windows

For CHATS7 May 2007 (31 days = 1488 timesteps at 30 min):
- **Spin-up:** First 5 days (days 121–125, DOY) — not included in loss, used to warm up soil temperature and canopy state.
- **Calibration window:** Days 126–145 (20 days = 960 timesteps).
- **Validation window:** Days 146–151 (6 days = 288 timesteps) — held out from gradient computation.

The 20-day calibration window is sufficient to observe a full range of radiation, VPD, and temperature conditions at CHATS7 (Bonan et al. 2025/2026 document that May 2007 spans unstable, neutral, and stable atmospheric regimes, covering the full dynamic range of turbulent fluxes).

### 3.3 Target variables and data quality filtering

**Primary targets:**
- Gross Primary Production (GPP): computed from model as `gppveg_canopy[p]` (same variable as `compute_gpp` in `diags/fd_grad_check.py`).
- Latent heat flux (LE): `lhflx_canopy[p]` — controls water balance and coupled to stomatal conductance.

**Secondary targets (for validation only in Phase 1):**
- Sensible heat flux (H).
- Friction velocity (u*) — sensitive to canopy turbulence parameterization.

**Filtering criteria (applied before loss computation):**
1. Remove nighttime periods (incoming SW < 5 W m⁻²) from GPP loss — nighttime GPP is near zero with high measurement uncertainty.
2. Remove periods with u* < 0.2 m s⁻¹ (weak mixing, flux footprint unreliable). The Bonan et al. (2025/2026) CHATS data screening identifies northerly wind periods and u* thresholds.
3. Remove precipitation periods (canopy interception changes energy balance in ways not currently calibrated).
4. For LE: remove periods where the observed energy balance closure ratio deviates by more than 20% from 1.0.

The fraction of retained half-hourly data for CHATS7 May 2007 is approximately 70% of daytime periods based on the filtering statistics reported in Bonan et al. (2025/2026), giving ~336 valid daytime timesteps in the calibration window.

---

## 4. Loss Function Design

### 4.1 Primary loss

The primary loss is a normalized multi-objective function combining GPP and LE:

```
L(θ) = w_GPP · RMSE_norm(GPP) + w_LE · RMSE_norm(LE) + λ · R(θ)
```

where:
```
RMSE_norm(X) = sqrt( mean( (X_model - X_obs)² ) ) / std(X_obs)
```

Normalization by the observed standard deviation ensures that GPP (typical daytime values 10–20 μmol CO₂ m⁻² s⁻¹) and LE (typical daytime values 150–350 W m⁻²) contribute equally to the loss regardless of units. Weights `w_GPP = 0.5`, `w_LE = 0.5` for equal contribution in Phase 1.

The normalized RMSE is equivalent to 1 − NSE^0.5 (where NSE is Nash-Sutcliffe efficiency) and is bounded below by 0.

### 4.2 Regularization

L2 regularization on log-transformed parameters relative to their default values:

```
R(θ) = sum_i ( log(θ_i / θ_i^default) )²
```

This penalizes departures from physically meaningful priors without preventing the optimizer from finding better values. The regularization strength `λ = 0.05` provides a weak pull toward defaults and can be tuned by examining whether optimized parameters leave the plausible range in synthetic tests.

### 4.3 Handling of model state and warm-up

Each call to `MLCanopyFluxes` operates on a single timestep given the model state `mlcanopy_inst`. The state threading during optimization requires:

1. **Warm-up phase:** Run the first 5 days without gradient computation using standard `MLCanopyFluxes` to bring soil temperature and canopy energy stores to quasi-equilibrium. This warm-up is run outside the optimization loop using the current parameter values (restarted at each outer iteration if parameters change substantially).

2. **Within-loop integration:** For each optimization step, run the calibration window sequentially, threading `mlcanopy_inst` state forward. This matches the pattern in `offline_executable/main.py` (`ModelAdvance` loop) but with parameters as JAX traced values.

3. **Gradient accumulation:** Accumulate the loss over all valid calibration timesteps before calling `.backward()` (i.e., `jax.grad` sees the summed loss).

### 4.4 Nighttime and missing data

- Nighttime GPP contribution: set to 0 (masking via `jnp.where(sw > 5, gpp_loss_t, 0.0)` to preserve differentiability).
- Missing observations: set the per-timestep loss contribution to 0 via the same masking pattern, not by removing timesteps from the JAX trace (which would cause retracing).
- LE outliers (energy balance closure > 20% error): mask identically. All masks are precomputed as static boolean arrays before JIT compilation.

---

## 5. Optimization Algorithm

### 5.1 Optimizer choice: Adam

Adam (Adaptive Moment Estimation) is the recommended optimizer for this experiment. Justification:

- **Heterogeneous parameter scales:** `vcmaxpft` is O(100) μmol m⁻² s⁻¹ while `g1_MED` is O(4) kPa^0.5. Adam's per-parameter adaptive learning rates normalize these differences without requiring manual scale tuning.
- **Noisy gradients:** The CLM-ML-JAX model includes numerical operations (bisection solver, stability-limited turbulence) that introduce plateau regions in parameter space. Adam's momentum terms smooth these.
- **Prior experience:** Aboelyazeed et al. (2023) used Adam with a learning rate of 0.045 and trained for ~600 iterations. Their framework recovered `Vcmax25` values within 10–15% of literature values.
- **L-BFGS alternative:** L-BFGS converges faster per iteration on smooth loss surfaces but requires line searches that are expensive under JAX JIT. Recommend Adam for Phase 1, L-BFGS as a refinement step after Adam converges to within 5% of optimum.

### 5.2 Learning rate schedule

- **Initial learning rate:** 0.01 (applied to log-transformed parameters, so this corresponds to ~1% multiplicative change per step on the original scale).
- **Warm restart schedule:** Cosine annealing with restarts every 50 steps, with minimum learning rate 0.001. This avoids getting stuck in local minima from premature convergence.
- **Patience-based reduction:** If loss does not decrease by > 0.1% over 20 steps, reduce learning rate by factor 0.5.

### 5.3 Batch strategy

**Full-sequence optimization** (all valid calibration timesteps per gradient step) is preferred over mini-batching because:

1. The CLM-ML-JAX model has strong temporal dependencies (soil moisture, canopy state). Mini-batches that break temporal continuity are physically inconsistent.
2. The calibration window (~336 valid daytime steps) fits entirely in GPU memory even for N=15 sites.
3. Aboelyazeed et al. (2023) used all available data for each gradient step (not mini-batches) and achieved stable convergence.

For multi-site Phase 3, run all sites in a single `jax.vmap` call and sum losses across sites before computing the gradient. This preserves the full-sequence nature while exploiting GPU parallelism.

### 5.4 Multi-site parallelism via vmap

The benchmark framework in `diags/benchmark_multisite.py` demonstrates that `jax.vmap` over the leading site dimension of `mlcanopy_inst` NamedTuple works without modifying any physics files, because `MLCanopyFluxes` in differentiable mode already uses concrete Python int `grid.ncan` and concrete patch index `p=1`. For parameter optimization:

```python
# Stack N sites: θ is a [N, n_params] array
def single_site_loss(theta_i, mlcanopy_inst_i, obs_i):
    # inject theta_i into mlcanopy_inst_i, run forward, compute loss
    ...

# vmap across sites
total_loss = jnp.mean(jax.vmap(single_site_loss)(theta, batched_inst, batched_obs))
grad_fn = jax.jit(jax.grad(lambda theta: total_loss_fn(theta)))
```

For PFT-specific parameters: θ has shape [n_pfts, n_params_per_pft]. Each site selects its PFT row via a static integer index (precomputed, not traced), which avoids dynamic indexing under vmap.

### 5.5 Convergence criteria

Stop when any of the following is met:
- Relative change in loss < 0.01% over 20 consecutive steps.
- Parameter changes < 0.1% (in log space) over 20 steps.
- Maximum 500 optimizer steps reached.
- Validation loss increases for 30 consecutive steps (early stopping against overfitting).

---

## 6. Implementation Plan

### 6.1 New files to create

**`diags/optimize_params.py`** — Main optimization script, structured analogously to `diags/fd_grad_check.py`:
```
diags/optimize_params.py
  - Imports: expt_init.py (shared model state initialization)
  - Defines: forward_loss(theta, obs_gpp, obs_le, masks)
  - Implements: Adam optimizer loop with logging
  - Outputs: optimized_params.json, loss_curve.png, parameter_trajectory.png
```

**`diags/expt_load_obs.py`** — Loads observed GPP and LE from tower NetCDF files, applies quality filters, and returns masked arrays aligned to model timesteps. Analogous to the data loading in `offline_driver/TowerDataMod.py` and `offline_driver/MLCanopyDataMod.py`.

**`diags/param_sensitivity.py`** — Computes the Jacobian `d(GPP, LE)/d(theta)` using `jax.jacobian` for all parameters at the default values. This identifies which parameters have non-zero gradients at the default operating point — a prerequisite sanity check before optimization.

**`bashscripts/run_optimize_params.sh`** — SLURM job script (A100 GPU, 8 hr walltime, 64 GB RAM). Template follows `bashscripts/run_multisite_benchmark.sh`.

### 6.2 Modifications to existing code

**`src/multilayer_canopy/MLpftconMod.py`** — Add a factory function `make_pft_params(theta_dict)` that constructs a `MLpftcon_type` NamedTuple from a dictionary of optimizable parameters, with all non-optimized fields at their default values. This is the injection point for the optimizer.

**`src/multilayer_canopy/MLCanopyFluxesMod.py`** — No physics changes required. The differentiable mode already supports parameter injection through `mlcanopy_inst` fields and `atm2lnd_inst` (as demonstrated in `diags/fd_grad_check.py`).

**`src/clm_src_biogeophys/MLCanopyFluxesType.py`** (or equivalent state type) — Verify that `vcmaxpft`, `iota_SPA`, and `g1_MED` are accessible as JAX-traced arrays through the state NamedTuple hierarchy. If they are stored as Python module globals in `MLpftconMod.py` (currently the case: `MLpftcon` singleton), the optimization loop must pass modified instances explicitly rather than relying on module-level state.

### 6.3 Training loop structure

```python
# Pseudocode for optimize_params.py

theta = init_theta()  # log-space, shape [n_params]
opt_state = adam_init(theta, lr=0.01)
obs_gpp, obs_le, masks = load_obs("CHATS7")

@jax.jit
def loss_and_grad(theta):
    pft_params = make_pft_params(jnp.exp(theta))  # log→linear
    total_loss = 0.0
    state = warm_up(pft_params, n_warmup_steps=240)  # 5 days @ 30min
    for t in range(n_cal_steps):
        state, gpp_t, le_t = forward_step(state, pft_params, forcing[t])
        loss_t = masked_loss(gpp_t, obs_gpp[t], le_t, obs_le[t], masks[t])
        total_loss += loss_t
    return total_loss + regularization(theta)

for step in range(500):
    loss, grads = jax.value_and_grad(loss_and_grad)(theta)
    theta, opt_state = adam_update(theta, grads, opt_state)
    log_step(step, loss, theta)
```

The critical insight from the `fd_grad_check.py` pattern: pass `mlcanopy_inst` as a function argument (not a module global) so JAX can trace through it. The `atm2lnd_inst._replace(...)` pattern used in `fd_grad_check.py` (lines 66–69) shows the correct idiom for injecting traced values into NamedTuple state.

### 6.4 Checkpointing

Save `theta`, `opt_state`, and `step` to disk every 50 optimizer steps using `np.savez`. Loss curve and parameter values at each checkpoint saved to `diags/figures/optimization_run_{timestamp}/`. This allows resuming interrupted runs and examining convergence trajectories.

---

## 7. Evaluation Plan

### 7.1 Calibration metrics (computed on training window)

- RMSE for GPP (μmol CO₂ m⁻² s⁻¹) and LE (W m⁻²)
- Pearson correlation coefficient R for GPP and LE
- Nash-Sutcliffe Efficiency (NSE) for both variables
- Mean bias (systematic offset)

Compare optimized vs. default CLM parameters on all metrics.

### 7.2 Validation metrics (computed on held-out window)

Same metrics computed on the held-out 6-day validation window (days 146–151). If optimized parameters degrade validation performance relative to calibration, this indicates overfitting.

### 7.3 Synthetic identifiability test

Before running on real data: generate synthetic observations by running the model with a known "true" parameter set (e.g., the `pftcon_val=1` CHATS values from `MLpftconMod.py` lines 311–316: `vcmaxpft=125, gplant_SPA=7, iota_SPA=375, root_resist_SPA=14, psi50_gs=-1.60`), then initialize optimization from the PFT-7 defaults and verify recovery. This is the "synthetic case" validation used by Aboelyazeed et al. (2023, their Section 3.1) who demonstrated near-perfect recovery of `Vcmax25` with 5% observational noise.

### 7.4 Comparison baseline

Three baselines:
1. Default CLM PFT-7 values (`pftcon_val=0`): `vcmaxpft=57.7, iota_SPA=750, g1_MED=4.45`
2. CHATS-tuned values (`pftcon_val=1`): `vcmaxpft=125, iota_SPA=375` (from Rosati et al. 2006 leaf gas exchange)
3. Literature ensemble: Aboelyazeed et al. (2023) Table 3 reports `Vcmax25` values for BET temperate (closest PFT to walnut) from CLM4.5 (62.5), AVIM (60), BETHY (58), and TRY database (61.1 μmol m⁻² s⁻¹)

### 7.5 Parameter uncertainty quantification

After finding the optimum, estimate parameter uncertainty via:
- **Laplace approximation:** Compute the Hessian `d²L/dθ²` using `jax.hessian` at the optimum. The inverse Hessian diagonal gives parameter variances. Practical for up to ~10 parameters.
- **Ensemble perturbation:** Run 50 optimization trials with different random initializations (within the prior bounds) and report the spread of converged values. This probes equifinality (multiple parameter combinations with similar loss).

---

## 8. Compute Estimate

### 8.1 Compilation cost

JIT compilation of the full 30-day forward pass with gradient tracing will take approximately 10–30 minutes on the A100 for the first call (CHANGELOG session 18 notes ~100s/step for single-site; session 20 confirms 6 RK sub-steps × 4 stages = 24 physics evaluations per 30-min timestep). Subsequent calls reuse the compiled XLA kernel. Plan for 1 initial compilation before the optimization loop.

### 8.2 Per-step cost

From CHANGELOG.md session 18:
- Single-site, per 30-min step (full physics RK4): ~100s (A100 GPU, highly under-utilized)
- vmap N=32, per step (Euler physics): 0.394s

For the full 31-day calibration window (~336 valid daytime steps):
- Single-site, full physics: ~336 × 100s ≈ 9.3 hours per gradient step (unacceptable)
- vmap N=15 sites, full physics: ~50–200s per gradient step (estimated from Euler scaling factor)

**Critical path:** The single-site slowdown identified in CHANGELOG.md session 18 (root cause: 46-element arrays give ~0% GPU occupancy) means full-physics optimization over 336 sequential timesteps is impractical without vmap. Two options:

**Option A (recommended):** Use vmap over a time-batch dimension within a single site. Stack 8 timesteps along the batch dimension, run them in parallel (approximation: treat each timestep as independent, ignoring within-batch state threading). This increases GPU occupancy by 8× at the cost of ignoring intra-batch state carry-over. Acceptable for parameters that control instantaneous fluxes (GPP, LE) rather than slow state variables (soil moisture).

**Option B:** Wait for the full-physics vmap N=32 benchmark results (SLURM job 7328915) to confirm throughput, then use vmap over the site dimension with 15 sites and run the full sequential time loop per site.

### 8.3 Total compute estimate

- Synthetic identifiability test: 100 optimizer steps × 2 min/step = ~3.5 hours
- Phase 1 (CHATS7 single site): 200 steps × ~5 min/step = ~17 hours
- Phase 2 (5 sites, vmap): 300 steps × ~8 min/step = ~40 hours

SLURM allocation recommendation: 48-hour jobs on A100 partition (as configured in `bashscripts/run_multisite_benchmark.sh`).

---

## 9. Risks and Mitigations

### 9.1 Gradient vanishing/exploding through long time series

**Risk:** Gradients propagated through 336 sequential timesteps of RK4 integration can vanish (if sensitivities attenuate) or explode (if local Jacobians > 1). The `alpha_tref` gradient explosion documented in CHANGELOG.md sessions 17–21 showed that even a single WUE bisection computation can produce O(10¹⁴⁴) gradients when the `bracket_ok` condition is mishandled.

**Mitigation:** 
1. Apply the IFT fix (session 21) for all bisection-based computations before beginning optimization.
2. Monitor gradient norms at each optimizer step; clip if `||grad|| > 1000` (gradient clipping).
3. Use gradient checkpointing (`jax.checkpoint`) for the RK4 substep loop to trade memory for recomputation, allowing longer sequences without memory overflow.
4. Start with short calibration windows (7 days) and extend after confirming gradient stability.

### 9.2 Local minima in parameter space

**Risk:** The loss surface `L(vcmaxpft, iota_SPA, g1_MED)` may have multiple local minima, particularly because GPP and LE can be similar under different combinations of high Vcmax25/low iota (Rubisco-limited, water-efficient) and low Vcmax25/high iota (light-limited, water-using).

**Mitigation:**
1. Run the synthetic identifiability test with 20 random initializations before real data optimization. If > 80% of trials converge to the same parameter set, the problem is identifiable.
2. Include both GPP and LE in the loss function — the joint constraint eliminates most equifinality because a parameter set that fits GPP with wrong transpiration is penalized.
3. Apply the L2 regularization term to prevent wandering far from the well-tested prior.

### 9.3 Equifinality

**Risk:** As noted by Aboelyazeed et al. (2023, Section 4), "within-PFT variation in Vcmax25 can be significant, and parameters could also be determined on the trait level as well as by multiple environmental factors." Multiple parameter combinations may fit observations equally well. For example, increasing `vcmaxpft` and decreasing `iota_SPA` by corresponding amounts can produce nearly identical GPP.

**Mitigation:**
1. Regularization toward priors breaks the symmetry for parameters with known independent constraints.
2. The CHATS dataset has within-canopy profile observations (wind speed, temperature, specific humidity) that constrain turbulence parameters independently of carbon fluxes (Bonan et al. 2025/2026, Figs. 9–13). In Phase 2, incorporate above-canopy H and u* into the loss to provide additional independent constraints.
3. Report the Hessian eigenspectrum: directions with near-zero curvature indicate equifinal combinations, which should be reported as unidentifiable rather than falsely precise.

### 9.4 Model structural error

**Risk:** Even at optimal parameters, the model may not reproduce observations due to structural limitations. For example, the CHATS paper (Bonan et al. 2025/2026) shows that strongly stable boundary layer conditions produce specific humidity biases of a few tenths of a g kg⁻¹ that the model cannot eliminate regardless of parameter values.

**Mitigation:**
1. Filter strongly stable periods (Richardson number Ri_b > 1) from the loss — these are regime-dependent errors, not parameter errors.
2. Report the minimum achievable loss (from the synthetic test with perfect parameters) as an estimate of the structural error floor.
3. Compare optimized-parameter residuals against the error patterns documented for the CHATS7 site in Bonan et al. (2025/2026) to distinguish parameter error from structural error.

---

## 10. Key Files Summary

| File | Role |
|---|---|
| `src/multilayer_canopy/MLpftconMod.py` | Source of all optimizable PFT parameters; add `make_pft_params()` factory |
| `src/multilayer_canopy/MLclm_varctl.py` | Model configuration; set `DIFFERENTIABLE_MODE=True` during optimization |
| `src/offline_driver/TowerDataMod.py` | Site metadata for all 15 AmeriFlux sites |
| `diags/fd_grad_check.py` | Template for forward/gradient computation pattern |
| `diags/expt_init.py` | Shared model state initialization; re-use in optimization script |
| `diags/optimize_params.py` | **[TO CREATE]** Main optimization script |
| `diags/expt_load_obs.py` | **[TO CREATE]** Observation loader with quality filtering |
| `diags/param_sensitivity.py` | **[TO CREATE]** Jacobian-based sensitivity analysis |
| `bashscripts/run_optimize_params.sh` | **[TO CREATE]** SLURM job script |

---

## References

- Aboelyazeed, D., Xu, C., Hoffman, F. M., Liu, J., Jones, A. W., Rackauckas, C., Lawson, K., and Shen, C. (2023). A differentiable, physics-informed ecosystem modeling and learning framework for large-scale inverse problems: demonstration with photosynthesis simulations. *Biogeosciences*, 20, 2671–2692.
- Bonan, G. B., Patton, E. G., Finnigan, J. J., and Baldocchi, D. D. (2021). Moving beyond the incorrect but useful paradigm: reevaluating big-leaf and multilayer plant canopies to model biosphere-atmosphere fluxes — a review. *Agricultural and Forest Meteorology*, 306, 108435.
- Bonan, G. B. and Burns, S. P. and Patton, E. G. (2025/2026). Beyond surface fluxes: Observational and computational needs of multilayer canopy models — A walnut orchard test case. *Agricultural and Forest Meteorology*, 378, 110960.
