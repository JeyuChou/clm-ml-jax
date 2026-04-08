# Experiment Plan: clm-ml-jax NeurIPS Paper
*Generated: 2026-04-08. Target submission: 2026-04-24 (AI4Science @ NeurIPS 2026)*

---

## 1. Paper Positioning

### Core Claim (one sentence)
We present a five-phase LLM-assisted agentic pipeline for translating legacy Fortran ESMs into validated, differentiable JAX implementations — demonstrated on CLM-ml-v2, a widely-used multilayer canopy model — and show that the resulting artifact enables scientific workflows (sensitivity analysis, gradient-based calibration) that were previously intractable.

### Target Track and Rationale
**Primary target: AI4Science workshop @ NeurIPS 2026** (4–6 pages).  
Rationale: 16 days to deadline is not enough time to produce a main-track NeurIPS paper. AI4Science is the correct venue for a validated software/methodology contribution with one or two concrete scientific demonstrations. Main NeurIPS track should be the target for a follow-on version with full calibration results and multi-site benchmarks.

**Do not submit to main NeurIPS track in April 2026.** A paper with zero actual experiment results in the body (all stubs) will be rejected. The workshop allows more honest "here is the tool + early results" framing.

### Three Most Comparable Prior Papers

| Paper | What they do | How clm-ml-jax differentiates |
|---|---|---|
| **JAX-CanVeg (Jiang et al., 2025, WRR)** | New differentiable LSM built natively in JAX; hybrid physics+DNN; tested at 4 AmeriFlux sites for LE/NEE | We faithfully *translate* an existing operational model (CLM-ml-v2) that thousands of users already depend on, preserving all physics and validating numerical equivalence. Our contribution is the *translation methodology* itself, not a new model. JAX-CanVeg cannot be used as a drop-in replacement for CLM-ml-v2 workflows. |
| **Aboelyazeed et al. (2023, Biogeosciences)** | Differentiable physics-informed ML for photosynthesis; learns Vcmax25 and water-stress parameters | They build a hybrid dPL model for one process (photosynthesis). We provide end-to-end gradients through the full multilayer canopy column including turbulence, radiation, stomata, and energy balance. |
| **Tang et al. (2025, AI-Researcher)** | Autonomous multi-agent scientific research system | We demonstrate a simpler but practical and reproducible agentic loop (the Ralph loop) for a specific high-value use case: ESM code translation. Our methodology is grounded in a real 28-module physics codebase with oracle validation. |

**Critical differentiation to make explicit in the paper:** JAX-CanVeg also has multilayer canopy capability. The key distinction is *faithfulness to an existing operational model*: clm-ml-jax is intended for users of CLM-ml-v2 who want to keep their parameterizations, their site configurations, their validation heritage — and gain differentiability as an add-on, not by switching to a different model. Frame it as "your existing model, now differentiable" not "a new differentiable model."

---

## 2. Current State Audit

### What exists and is credible today (April 8, 2026)
- **Oracle validation**: Formally done and passing. Scatter plots, flux comparisons, and profile comparisons on disk (`diags/figures/validation_flux.png`, `validation_profiles.png`, `validation_scatter.png`). These figures need to be described in the paper and their pass criteria stated explicitly.
- **Finite gradients**: Confirmed via `diags/quick_grad_check.py`. All four output classes (tair_profile, eair_profile, tleaf_leaf, tg_soil) produce finite gradients. Max absolute gradient values are large (7.9e5 for tair_profile) but finite — this is expected given unnormalized inputs/outputs.
- **Translation methodology**: Fully implemented and described in the draft (Phase 1–5, Ralph loop, CLAUDE.md, plan.md, activity.md, CHANGELOG.md). This is a genuine contribution.
- **Internal compilation improvements**: 13.2× speedup in steady-state from eliminating XLA recompilations (baseline → lru_cache → fori_loop fixes). Useful engineering story but NOT a speedup vs. Fortran.

### What is broken, missing, or unverified

| Claim in draft | Reality | Action needed |
|---|---|---|
| "Significant runtime improvements via JAX JIT" | Steady-state is **120s/timestep** on V100S at 75% occupancy. No Fortran timing has been run. This is not a speedup claim — it may actually be slower than Fortran. | **Either run a Fortran comparison and report honestly, or remove the speedup claim entirely.** Do not claim speedup without data. |
| "JAX+JIT+vmap (batched over multiple sites)" | vmap over multiple sites is not implemented. | Remove from Experiment 2 or mark as future work. |
| Experiment 3 (gradient-based calibration) | Not implemented. No calibration loop exists. | Implement a minimal demo (see Experiment 3 below). |
| Experiment 4 (Jacobian sensitivity analysis) | Not implemented but feasible. `jax.jacfwd` can be called once gradients work. | Implement (see Experiment 4 below). |
| "Numerically identical outputs" | "Numerically identical" is too strong. The draft also says rtol=1e-5. Use "validated to rtol=1e-5" consistently. | Fix language in abstract and Section 4.2. |
| Title: "Neurosymbolic Agentic AI pipeline" | "Neurosymbolic" means hybrid neural-symbolic architectures. This is not that. | **Rename the paper.** See suggestion below. |

### Suggested title
*"clm-ml-jax: An LLM-Assisted Agentic Pipeline for Translating Legacy Fortran Land Surface Models into Differentiable JAX"*

---

## 3. Experiment Roadmap

### MUST-HAVE (paper cannot be submitted without these)

---

**Experiment 1: Oracle Validation**

- **Scientific question**: Does clm-ml-jax reproduce CLM-ml-v2 Fortran outputs to within acceptable numerical tolerance?
- **Why a reviewer cares**: This is the credibility gate. Without it, all downstream claims (differentiability, calibration) are unverifiable.
- **Implementation**: Already done. Need to write it up properly.
- **Data**: CHATS7 site, 2007-05 forcing data (already used for validation).
- **Metrics**: Max absolute error and rtol per output variable (LE, H, NEE, canopy temperature profile, stomatal conductance). Report per-variable across all timesteps.
- **Expected result**: All variables within rtol=1e-5 (already achieved). Table 1 of the paper.
- **Effort**: Low — results exist. 1 day to write up and format Table 1.
- **Priority**: Critical

---

**Experiment 2: Finite Gradients + Gradient Correctness**

- **Scientific question**: Does autodiff through the full CLM-ml-jax column produce correct, finite gradients?
- **Why a reviewer cares**: This is the foundation of the differentiability claim. JAX-CanVeg also has gradients — showing yours work and are correct is table stakes.
- **Implementation**: `diags/quick_grad_check.py` already confirms finite gradients. Need to add a finite-difference gradient check for 2–3 parameters (e.g., Vcmax25, Ball-Berry slope `m`) to confirm autodiff is accurate, not just finite.
- **Data**: Single CHATS7 timestep (same as oracle validation).
- **Metrics**: Max relative error between jax.grad and finite-difference gradient for the checked parameters. Target: <1% agreement.
- **Effort**: Low-Medium — 1–2 days to implement finite-difference check and run comparison.
- **Fallback**: If finite-difference check fails for the full model, restrict to a single module (leaf photosynthesis). Still a valid result.
- **Priority**: Critical

---

### SHOULD-HAVE (significantly strengthen the story)

---

**Experiment 3: Sensitivity Analysis via Jacobians**

- **Scientific question**: Which physiological parameters most influence surface energy and carbon fluxes?
- **Why a reviewer cares**: This is the most concrete scientific result enabled by differentiability. It demonstrates a capability that is genuinely unavailable with the Fortran model and directly addresses a long-standing scientific question (which parameters matter most in CLM-ml-v2?).
- **Implementation**:
  ```python
  # Compute full Jacobian of [LE, H, NEE] w.r.t. parameter vector
  J = jax.jacfwd(forward_fn)(params)  # shape: [n_outputs, n_params]
  # Plot as heatmap
  ```
  Parameter set: 5–8 physiological parameters (Vcmax25, Ball-Berry slope m, g0, leafc, clumping factor, ...). Choose parameters with clear physical interpretation.
- **Data**: CHATS7 site, single representative timestep (midday, peak photosynthesis).
- **Metrics**: Normalized Jacobian heatmap (∂output/∂param, normalized by output std). Runtime comparison: jacfwd call time vs. estimated finite-difference cost (N_params × 1 forward call).
- **Expected result**: Clear sensitivity structure (e.g., LE most sensitive to Ball-Berry slope, NEE most sensitive to Vcmax25). Jacobian computed in ~1 forward-pass-equivalent time vs. N_params × 1 forward pass for finite differences.
- **Effort**: Medium — 2–3 days. Requires a clean `forward_fn(params)` wrapper that takes a parameter dict and returns scalar fluxes. The gradient machinery already works; this is mostly plumbing.
- **Fallback**: If full-model jacfwd is too slow (>hours), restrict to a single canopy layer or a short 1-timestep run. Still demonstrates the capability.
- **Priority**: High

---

**Experiment 4: Simple Parameter Calibration Demo**

- **Scientific question**: Can gradient-based optimization recover a known parameter from simulated flux tower data faster than gradient-free search?
- **Why a reviewer cares**: This is the "killer app" of differentiability for LSM calibration. Aboelyazeed et al. (2023) did this for photosynthesis only; demonstrating it for the full column is a genuine advance.
- **Implementation**:
  1. Fix ground truth: run CLM-ml-jax with known Vcmax25 value to generate synthetic "observations" of LE and NEE.
  2. Perturb Vcmax25 by ±30% as starting point.
  3. Run gradient descent (Adam, 50–100 steps) to recover the true value.
  4. Run Nelder-Mead or random search baseline with equal computational budget (same number of forward calls).
  5. Compare: convergence speed, final RMSE, number of forward evaluations.
- **Data**: Synthetic observations from CHATS7 run (no real data needed — cleaner result).
- **Metrics**: Parameter recovery error vs. number of forward evaluations. Final RMSE on held-out timesteps.
- **Expected result**: Adam converges to <5% parameter error in 50 steps (~50 forward-pass equivalents). Nelder-Mead requires 200–500 evaluations for comparable accuracy.
- **Effort**: Medium-High — 4–5 days. Key engineering challenge: wrapping the forward pass to accept a parameter dict cleanly (separate from state initialization). This is the riskiest experiment given the timeline.
- **Fallback**: If full-column calibration is too slow per step, calibrate only the leaf photosynthesis submodule in isolation. Still a valid proof-of-concept.
- **Priority**: High — but do Experiment 3 first. If you run out of time, Experiment 3 alone is sufficient for a workshop paper.

---

### NICE-TO-HAVE (include only if time permits)

---

**Experiment 5: Runtime Comparison (Honest Framing)**

- **Scientific question**: How does CLM-ml-jax wall-clock time compare to the Fortran reference?
- **Why a reviewer cares**: Reviewers will ask. Better to have honest numbers than to be caught overclaiming.
- **Current reality**: 120s/timestep on V100S at 75% occupancy. This is almost certainly *slower* than Fortran. The internal optimization story (1942s → 120s by eliminating XLA recompilations) is an engineering contribution, but it is not a speedup vs. Fortran.
- **Honest framing if you run it**: "CLM-ml-jax achieves parity with the Fortran reference on CPU/GPU. Further optimization (kernel fusion, vmap across sites) is left for future work. The primary performance gain is in gradient computation, which has no Fortran equivalent."
- **Effort**: Low — 1 day to run Fortran on the same hardware. But only do this if you have 2+ days to spare.
- **Priority**: Optional. Do NOT include a runtime table unless you have Fortran comparison numbers.

---

## 4. Figures Plan

| # | Figure | Data source | Experiment | Status |
|---|---|---|---|---|
| 1 | Oracle scatter: JAX vs Fortran (LE, H, NEE) | `validation_scatter.png` | Exp 1 | **Exists on disk** — needs caption and write-up |
| 2 | Oracle time series: flux comparison, 1 day | `validation_flux.png` | Exp 1 | **Exists on disk** — needs caption and write-up |
| 3 | Oracle vertical profiles: temperature, humidity, CO2 | `validation_profiles.png` | Exp 1 | **Exists on disk** — needs caption and write-up |
| 4 | Gradient finite-difference check: jax.grad vs FD for 2–3 params | New | Exp 2 | Not yet |
| 5 | Jacobian heatmap: ∂(LE,H,NEE)/∂(parameter set) | New | Exp 3 | Not yet |
| 6 | Calibration convergence: loss vs. evaluations (Adam vs. baseline) | New | Exp 4 | Not yet |
| 7 | Methodology diagram: five-phase pipeline + Ralph loop schematic | Schematic | N/A (Methods figure) | Not yet |

**Minimum viable figure set for workshop submission: Figures 1, 2, 5, 7** (oracle validation + sensitivity analysis + methodology diagram). This is achievable in 16 days.

---

## 5. Timeline (April 8 → April 24)

**Total: 16 days. No buffer weeks — this is already a compressed schedule.**

### Week 1: April 8–14 (Infrastructure + Core Results)

| Day | Task | Deliverable |
|---|---|---|
| Apr 8–9 | Fix paper draft: remove speedup claims, fix "neurosymbolic" title, write oracle validation section using existing figures | Sections 2, 3, 4.1, 4.2 drafted with real numbers |
| Apr 9–10 | Implement finite-difference gradient check for Vcmax25 and Ball-Berry m | Figure 4 + pass/fail table |
| Apr 10–12 | Implement `forward_fn(params)` wrapper for jacfwd; run Jacobian over 5–8 parameters | Figure 5 (Jacobian heatmap) |
| Apr 12–13 | Make methodology diagram (Figure 7): five phases + Ralph loop | Figure 7 |
| Apr 13–14 | Write Sections 3 (Methodology) and 4.3 (Sensitivity analysis) | Draft complete for core sections |

**Go/no-go decision point (April 14):** If Jacobian heatmap is compelling and gradient check passes, the paper has enough for a workshop submission. Proceed to calibration demo. If jacfwd is too slow or numerically noisy, skip Experiment 4 and move directly to writing.

### Week 2: April 15–21 (Calibration Demo + Writing)

| Day | Task | Deliverable |
|---|---|---|
| Apr 15–17 | Implement synthetic calibration demo (Exp 4): Adam vs. Nelder-Mead on Vcmax25 | Figure 6 + calibration table |
| Apr 17–19 | Write Sections 4.4 (calibration), 5 (Discussion), 6 (Conclusion), Related Work | Full draft complete |
| Apr 19–20 | Integrate all figures; ensure all claims in text are backed by figures/tables | Complete draft with no TODOs in the body |
| Apr 20–21 | Co-author review pass; fix remaining citations ([CITE] stubs) | Submission-ready draft |

**Go/no-go decision point (April 17):** If calibration demo is not converging cleanly, cut it and use the remaining time for writing. A clean sensitivity analysis (Exp 3) + oracle validation (Exp 1) + gradient check (Exp 2) is sufficient for AI4Science.

### Final stretch: April 22–24

| Day | Task |
|---|---|
| Apr 22 | Final proofread; verify all figures render correctly in PDF |
| Apr 23 | Checklist completion (checklist.tex in repo) |
| Apr 24 | **Submit** |

---

## Appendix: Claims to Remove or Fix Before Submission

These claims appear in the current draft and are unsupported. Fix before submission:

1. **"Achieves significant runtime improvements via JAX JIT compilation"** (abstract) — Remove or replace with "eliminates recompilation overhead" unless Fortran comparison data exists.
2. **"JAX+JIT+vmap (batched over multiple sites)"** (Exp 2) — Remove. Not implemented.
3. **"Numerically identical outputs"** — Replace with "validated to rtol=1e-5" everywhere.
4. **"Neurosymbolic Agentic AI pipeline"** (title) — Wrong terminology. See suggested title above.
5. **"Full Jacobian at the cost of a single forward pass per output dimension"** — Technically accurate for jacrev, but jacfwd is one pass per input dimension. Clarify which you use and why (forward mode is cheaper when n_params < n_outputs, reverse mode otherwise).
6. **Missing citations**: The Related Work section has 8+ `[CITE]` stubs. These must be resolved. At minimum, cite: Bonan (CLM-ml-v2), Bradbury et al. (JAX), Jiang et al. (JAX-CanVeg), Aboelyazeed et al. (dPL/δpsn), Bonan et al. 2021 (multilayer vs big-leaf), Sharma et al. (clax) if it exists.
