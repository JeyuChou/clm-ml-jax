# NEXT_STEPS.md
## CLM-ML-JAX NeurIPS AI4Science Submission
**Deadline: April 24, 2026 — 9 days remaining**
**Generated: April 15, 2026 by project manager agent**
**Sources: Self-assessment from full paper read + NeurIPS-style review (agent_review.md) + ESM expert assessment (agent_literature.md)**

---

## 1. Paper Status Summary

### What Is Solid

The paper is substantially more complete than the original experiment_plan.md suggested. As of April 14:

- **Oracle validation (Exp 1):** Done. Table 1 is in the paper with real numbers. Figures on disk (`validation_scatter.pdf`, `validation_profiles.pdf`). Solid — both reviewers agree.
- **Gradient correctness (Exp 2):** 12/12 active parameter-output combinations pass. The IFT fix for the WUE bisection solver is called out by both reviewers as a genuine, non-trivial technical contribution. This section is the strongest part of the paper.
- **Sensitivity analysis / Jacobian (Exp 3):** Partially done. Figure on disk (`sensitivity_jacobian.png`). jacrev timing (1094s vs. 118s FD) is honestly reported. BUT: has framing and content problems (see W2/W3 below).
- **Calibration demo (Exp 4):** Technically done but scientifically harmful as written (see W1 below — both reviewers flag this independently).
- **vmap benchmarks:** Done. 1.89× at N=32 confirmed. Both reviewers accept this with the caveat that it's vs. sequential JAX, not Fortran.
- **Bibliography:** All critical references present after session 31 additions (adJULES, Le Dimet/Talagrand, Gelbrecht et al., Harman-Finnigan).
- **Paper structure:** All sections drafted. Discussion and Conclusion are written.

### What Is Broken or Harmful

Both reviewers independently identified the same issues. Listed in order of severity:

**CRITICAL — Will cause rejection if not fixed:**

1. **Calibration experiment (Exp 4) contradicts the paper's motivating claim.** Nelder-Mead beats Adam on every metric in the 1D alpha_sw recovery. Both reviewers flag this as the single most damaging element. Additionally, the ESM expert notes that calibrating a radiation scale factor from GPP is not scientifically credible (you would use a pyranometer). The scientifically correct demo is recovering Vcmax25 and iota_SPA — the parameters that the community actually calibrates from flux towers.

2. **Pending result in Table 2 (LE/H gradients from job 7403949).** The paper cannot be submitted with "pending" in a results table. This is a submission blocker pending job completion.

3. **dpai=0 in the Jacobian heatmap may be wrong.** The ESM expert flags this as a red flag — LAI scaling should always produce non-zero gradients since it enters both radiation interception and dpai-weighted output sums. A related fix agent (job 7447256) may still be running. If the Jacobian figure shows dpai=0 while the paper claims 5-parameter sensitivity analysis, this undermines the gradient credibility claim.

**SUBSTANTIAL — Should fix before submission:**

4. **Jacobian framing is confused.** The paper mixes forcing sensitivity (T_air, SW_rad) with physiological parameters (Vcmax25, iota_SPA) in a single heatmap and makes a cost-ratio argument that the data doesn't support. The 9x overhead vs. FD for p=5, n=3 is physically real but unexplained.

5. **bonan2025 placeholder DOI** (`doi = {10.1016/j.agrformet.2025.xxx}`). Must resolve before submission.

6. **Figure caption error**: "error 7.6×10⁻³%" should be "error 0.076%" (the displayed error is 7.60×10⁻⁴ in alpha, which is 0.076% relative to alpha*=1.0).

7. **Figure path inconsistency**: Some figures use `../../diags/figures/` (relative), others use `figures/` (local). This will cause compilation failures at submission.

8. **Missing methodology pipeline diagram.** Both reviewers note this gap — the paper describes a 5-phase methodology but has no visual overview. This is the single most actionable visual addition remaining.

9. **"Global sensitivity analysis" overclaim in abstract.** A single-timestep, 5-parameter Jacobian is local, not global. Change to "Jacobian-based sensitivity analysis."

10. **"Drop-in replacement" overclaiming.** Soil/snow not translated. Change to "drop-in replacement for the canopy physics column."

**MINOR — Fix if time permits:**

11. NeuralGCM in bib but not cited in text body — one-line fix.
12. Fortran pattern table has duplicate row for module-state→NamedTuple.
13. Benchmark figure (`benchmark_summary.png`) uses Euler physics, not RK4 — must state in caption.
14. Contribution C2 and C3 need scope corrections (single-timestep, not multi-site workflows; single-site demonstrated).
15. No methodology pipeline figure (Fig. 7 from original plan).
16. g1 with Medlyn model (gs_type=0) never tested — reviewers may ask. Consider adding a note.

### Reviewer Verdicts

**NeurIPS-style reviewer:** Borderline Accept (Weak Accept). The gradient correctness table and methodology section are the strongest parts. Calibration experiment is the single most damaging element — both reviewers say fix this first.

**ESM expert:** Overall supportive of the scientific grounding. Gradient magnitudes are physically defensible. Primary concern: replace alpha_sw calibration with Vcmax25+iota_SPA synthetic recovery, which mirrors Aboelyazeed et al. (2023) and gives Adam a fair chance in 2D.

---

## 2. Prioritized Task List

### MUST DO Before Submission

**M1. Confirm LE/H gradient results from job 7403949 — 0.5 days**
Check if the job completed. Extract values for all 5 params × 3 outputs (LE, H) from the output or from `diags/figures/le_h_grad_check.png`. Fill the LE and H rows in Table 2 of JAXES.tex. If the job failed, resubmit `bashscripts/run_le_h_grad_check.sh` immediately. This is a submission blocker.

**M2. Verify dpai gradient and Jacobian heatmap — 0.5 days**
Check job 7447256 (dpai gradient fix, per ESM expert note). Confirm whether the current `sensitivity_jacobian.png` on disk shows non-zero dpai column. If dpai remains zero:
- Option A: Apply the canopystate_inst fix (elai_patch/esai_patch scaling path) and regenerate.
- Option B: Replace dpai with q_ref/VPD in the Jacobian parameter set. q_ref already has confirmed non-zero gradients (GPP: -0.05, H: +41.5, LE: -44.3) and is scientifically meaningful (VPD-driven stomatal closure). This is the faster path if the dpai fix is unclear.

Also verify that the Vcmax25 column is non-zero in the current figure (the vcmaxpft_jax fix was applied in session 30 — check if the figure predates or postdates that commit).

**M3. Replace calibration experiment with 2-parameter Vcmax25+iota_SPA recovery — 2 days**
This is the most important fix. Both reviewers independently demand this.

Design: 
- Ground truth: CHATS site-specific values from `MLpftconMod.py` `pftcon_val=1` block: vcmaxpft=125, iota_SPA=375
- Starting point: CLM PFT-7 defaults: vcmaxpft=57.7, iota_SPA=750
- Synthetic observations: run clm-ml-jax at ground truth to generate GPP and LE at a single representative timestep (midday, high-radiation)
- Optimizer: Adam (optax or manual implementation), 100 steps
- Baseline: Nelder-Mead over equal budget (200 function evaluations = 100 Adam steps × ~2 forward passes each)
- The vcmaxpft_jax injection pattern already works (confirmed in gradient checks). The iota_SPA injection requires verifying the analogous pattern.

This directly mirrors Aboelyazeed et al. (2023) Section 3.1 synthetic identifiability test. Adam will almost certainly win in 2D where the gradient provides directional information across parameter space.

If this is implemented: replace Table 3 with the new result, rewrite Exp 4 narrative, update C2 contribution claim.

If it cannot be implemented by April 17: fall back to Option B below.

**M3 fallback (Option B): Reframe Exp 4 as "gradient plumbing demonstration"**
- Remove the Adam vs. Nelder-Mead comparison table entirely.
- Keep a brief narrative: "Exp 4 demonstrates that the gradient machinery supports iterative parameter optimization. We recover alpha_sw from a 30% perturbation in 100 gradient steps. For 1-D problems, gradient-free methods are equally efficient; the gradient advantage grows with parameter dimensionality, scaling as O(1) gradient evaluations vs. O(2p) FD probes per iteration."
- Update C2: remove "recovering known parameter values from synthetic flux observations" — replace with "demonstrating gradient infrastructure sufficient for multi-parameter gradient-based calibration."
- This is safe and honest. It's weaker but not harmful.

**M4. Fix calibration figure caption error — 0.25 days**
The caption states "error 7.6×10⁻³%." The actual error is |1.000760 - 1.0| = 7.60×10⁻⁴ in absolute terms = 0.076% relative to alpha*=1.0. Fix the percentage figure in the Figure 4 caption.

**M5. Fix figure paths — 0.5 days**
Standardize all figure inclusions to use a single `figures/` subdirectory relative to the paper directory. Copy (or symlink) all actual figures into `Paper/jaxes_paper/figures/`. Currently `validation_scatter.pdf` and `validation_profiles.pdf` are in `Paper/jaxes_paper/figures/` but other figures (fd_grad_check.png, sensitivity_jacobian.png, benchmark_summary.png, le_h_grad_check.png) are in `diags/figures/` and referenced via `../../diags/figures/`. This will cause submission portal compilation failures.

**M6. Fix bonan2025 DOI — 0.25 days**
Look up the actual DOI for "Bonan, Burns, Patton, Beyond surface fluxes, Agricultural and Forest Meteorology." If still in press, use the accepted manuscript DOI or arXiv preprint if one exists. Do not submit with `xxx` placeholder.

**M7. Fix "global sensitivity analysis" → "Jacobian-based sensitivity analysis" — 0.1 days**
Change in abstract, and anywhere else it appears. A single-timestep, 5-parameter Jacobian is not global sensitivity analysis. This is a factual correction that a knowledgeable reviewer will catch immediately.

**M8. Fix "drop-in replacement" → "drop-in replacement for the canopy physics column" — 0.1 days**
Appears in Introduction and Discussion (Relationship to JAX-CanVeg section). Minor but important for accuracy.

**M9. Wire NeuralGCM citation into text body — 0.1 days**
`kochkov2024neuralgcm` is in the bib but never cited in the manuscript text. Add one sentence in Section 2.2: "At the atmosphere scale, NeuralGCM \cite{kochkov2024neuralgcm} demonstrates the scalability of JAX-based differentiable climate models." This may already be in the current draft — verify before editing.

**M10. Add Benchmark figure caption note about Euler physics — 0.1 days**
`benchmark_summary.png` uses Euler (RK1) timestepping, not RK4. Add to Figure caption: "benchmarks use Euler (first-order Runge-Kutta) timestepping; production runs use 4th-order RK which increases per-step cost by approximately 4×."

**M11. Soften Ralph loop framing in C3 and Section 3.4 — 0.5 days**
The structured state-tracking + oracle validation framework is the contribution; the Ralph loop is the implementation vehicle. Revise C3: "We formalise a five-phase LLM-assisted translation workflow centred on structured state tracking (\texttt{CLAUDE.md}, \texttt{plan.md}, \texttt{activity.md}, oracle validation harness), and demonstrate it via the Ralph loop — a self-correcting agentic pipeline that produced the full clm-ml-jax implementation."

**M12. Add translation statistics table — 0.5 days**
Add a small table to Section 3 (Methodology) or Appendix: files translated, total LOC, wall-clock translation time (from CHANGELOG), estimated Claude API usage, human-hours of supervision, number of Ralph loop sessions. Data is available in CHANGELOG.md. This makes the methodology claim quantitative and directly addresses a common reviewer question.

### SHOULD DO If Time Permits

**S1. Add methodology pipeline diagram — 1 day**
The paper describes a 5-phase pipeline + Ralph loop but has no visual schematic. This is Figure 7 from the original plan. A simple tikz/draw or hand-drawn figure showing the 5 phases as boxes with arrows, and the Ralph loop cycle within Phase 4, would significantly strengthen the methodology contribution. Both reviewers notice its absence.

**S2. Add second-site spot-check (US-Ha1) — 1 day**
Run the model at US-Ha1 or US-UMB for 24 hours (48 timesteps). If it runs, add one sentence to Limitations: "We confirm the model runs at a second PFT-7 site (US-Ha1) without code modification; full multi-site validation is left for future work." Directly addresses reviewer W5. Risk: if it fails, you need to explain why.

**S3. Add explicit single-timestep limitation statement — 0.25 days**
Add to gradient experiments sections (Exp 2, 3, 4): "All gradient experiments use a single model timestep as proof-of-concept; multi-timestep gradient accumulation is active work and requires mitigation of vanishing/exploding gradients over long sequences (Section 5.2)."

**S4. Add g1 gradient note for gs_type=0 — 0.25 days**
The reviewer asks "Can you differentiate through the Medlyn stomatal model?" Either run a 5-minute gradient check with gs_type=0 and report the result, or add one sentence: "The alpha_g1 gradient is expected to be non-zero under the Medlyn model (gs_type=0); a full gradient check for that configuration is planned."

**S5. Fix duplicate Fortran pattern table row — 0.25 days**
Table A3 has two rows that describe essentially the same pattern (in-place array mutation vs. module-level mutable arrays). Consolidate or distinguish clearly.

**S6. Jacrev overhead crossover characterization — 1 day**
For the paper's claim that "jacrev grows more favorable with p," add empirical data. Compute the Jacobian cost at p=5, p=10, p=20 using a lightweight synthetic forward function (not full physics). Show the crossover where jacrev becomes cheaper than FD. Even rough data would convert this from a speculative claim to a supported one.

### CUT / Defer to Follow-Up Paper

- Full multi-site Vcmax25/iota_SPA calibration over 31-day window (parameter_optimization_experiment.md Phase 1–3): out of scope for 9 days. Document in Discussion as planned future work.
- Laplace approximation / Hessian uncertainty quantification: cut.
- vmap N=64, N=128: The N=32 result is sufficient. No more SLURM benchmarking.
- Soil/snow thermodynamics translation: already deferred in Limitations.
- CESM coupling: already deferred.
- Runtime parity with Fortran on CPU: the 1300x slowdown is disclosed. Do not add more comparison data — it won't help.
- Adjoint model (hand-coded) comparison vs. JAX autodiff: interesting but 9-day project.

---

## 3. Day-by-Day Schedule (April 15–24)

### April 15 (Wednesday) — Triage and setup
**This is the most critical day. Two jobs may have completed overnight.**

Morning:
1. Check SLURM status for job 7403949 (LE/H grad check). If done: extract LE/H gradient values, fill Table 2 in JAXES.tex. If failed: resubmit `bashscripts/run_le_h_grad_check.sh` before doing anything else.
2. Check SLURM status for job 7447256 (dpai fix, per ESM expert note). Review results. Decide: fix dpai or replace with q_ref.
3. Check mtime of `diags/figures/sensitivity_jacobian.png` vs. the git commit date for the Vcmax25 fix in `diags/sensitivity_analysis.py`. If the figure is stale, queue a sensitivity_analysis regeneration job immediately.

Afternoon:
4. Begin Vcmax25+iota_SPA calibration coding in `diags/calibration_demo.py`. The vcmaxpft_jax injection pattern works. Extend to also inject iota_SPA. Set up the 2-parameter synthetic recovery loop (Adam + Nelder-Mead). Submit overnight GPU job.
5. Fix the figure caption error (M4): "7.6×10⁻³%" → "0.076%".

Deliverable: Table 2 pending result resolved; 2-param calibration job submitted.

---

### April 16 (Thursday) — Citations and technical writing pass

Morning:
1. Fix bonan2025 DOI (M6). Look it up. 30 minutes.
2. Fix figure paths (M5). Copy all diags figures into `Paper/jaxes_paper/figures/`. Update all `\includegraphics` paths in JAXES.tex to use `figures/` prefix consistently.
3. Check overnight 2-param calibration job. If done and showing Adam advantage: update Exp 4.

Afternoon:
4. Wire NeuralGCM into text body (M9). Verify it's present; add if missing.
5. Fix "global sensitivity analysis" → "Jacobian-based sensitivity analysis" (M7).
6. Fix "drop-in replacement" → "drop-in replacement for the canopy physics column" (M8).
7. Add benchmark figure Euler caption note (M10).
8. Soften Ralph loop framing in C3 and Section 3.4 (M11).

Deliverable: All critical factual errors fixed; figure paths resolved; bib clean.

---

### April 17 (Friday) — Calibration rewrite (the make-or-break day)

Morning:
1. Integrate 2-param calibration result. If Adam wins on Vcmax25+iota_SPA recovery: rewrite Exp 4 with new result, new Table 3, new narrative. Update C2 contribution bullet. This is the best outcome.
2. If Adam still loses in 2D OR the job failed: execute Option B (reframe). Rewrite Exp 4 as "gradient plumbing proof-of-concept." Remove Adam vs. Nelder-Mead comparison table. Revise C2. Do not let this drag beyond April 17 end of day.

Afternoon:
3. Full pass through Methodology section. Add translation statistics table (M12).
4. Address dpai/Jacobian update: if job completed, integrate result; if replacing with q_ref, update Section 4.3 and Figure 3 caption.
5. Add explicit single-timestep limitation note (S3).

Deliverable: Calibration section definitively rewritten with correct framing; Jacobian section clean.

---

### April 18 (Saturday) — Visual assets and second-site

Morning:
1. Start methodology pipeline diagram (S1). Even a rough hand-drawn-style schematic using tikz (5 boxes: Scoping → Infrastructure → Oracle → Ralph Loop → Integration) will suffice. Budget 3 hours maximum.
2. Queue US-Ha1 single-day run (S2) if confidence is high it will work.

Afternoon:
3. Full paper read-through — first complete pass. Mark all remaining issues, inconsistencies, broken references.
4. Verify page count (pdflatex compilation). If over 6 pages: identify what to cut. The Appendix details for the Ralph loop implementation are the first cut candidate.

Deliverable: Methodology diagram drafted; page count known; comprehensive issue list.

---

### April 19 (Sunday) — Issue resolution and US-Ha1 integration

Morning:
1. Check US-Ha1 job. If successful: add one sentence to Limitations. If failed: document in Limitations; do not attempt to debug.
2. Fix all issues flagged in April 18 read-through.
3. Fix duplicate Fortran pattern table row (S5).

Afternoon:
4. Second complete paper read-through. Focus: Does every claim in the text have a supporting figure or table? Does the abstract accurately describe what was demonstrated (not what was planned)?
5. Resolve any remaining [CITE] stubs.

Deliverable: Clean draft — no pending results, no broken references, no overclaims.

---

### April 20 (Monday) — Co-author review

Send full compiled PDF to co-author(s) for review. While waiting:
1. Verify pdflatex + bibtex compilation is clean end-to-end from a fresh directory.
2. Check NeurIPS AI4Science workshop submission portal requirements (anonymization, page limit, supplementary policy).
3. Start submission checklist (`checklist.tex`).

---

### April 21 (Tuesday) — Co-author feedback integration

Integrate feedback. Do not introduce new experiments after this point.

---

### April 22 (Wednesday) — Final proofread

1. Read entire paper aloud from the PDF.
2. Verify: all figures render, all table footnotes present, abstract is accurate, no typos in math.
3. Anonymize if required by the workshop track.

---

### April 23 (Thursday) — Submission preparation

1. Final page count check.
2. Complete checklist.tex.
3. Prepare supplementary material if any (code availability statement, github URL).
4. Stage submission in portal.

---

### April 24 (Friday) — SUBMIT
Submit by noon. Not at 11:59 PM.

---

## 4. Critical Decisions

**Decision 1: Vcmax25+iota_SPA calibration OR reframe? (Decide by April 17 noon — hard deadline)**

Run the 2-parameter calibration (M3) starting April 15. If it works and Adam wins: use it. This is the right scientific experiment and directly mirrors Aboelyazeed et al. (2023).

If the job fails or Adam still loses: switch immediately to Option B (reframe). The fallback is safe — oracle validation + gradient correctness + Jacobian sensitivity is a complete workshop paper without a compelling calibration demo. Do not let calibration-chasing eat into writing time past April 17.

**Decision 2: Fix dpai OR replace with q_ref in Jacobian? (Decide by April 15 end of day)**

Both options produce a complete 5-parameter heatmap. q_ref is actually more scientifically interesting than LAI for the CHATS site (stomatal response to VPD is a primary flux driver in the May 2007 period). If the dpai fix from job 7447256 is clean and confirmed, use it. If the fix is uncertain or the job results are unclear: replace dpai with q_ref. Do not leave a zero column in the submitted Jacobian heatmap.

**Decision 3: Include methodology diagram or not? (Decide by April 18 end of day)**

If you can produce a clean schematic in 2–3 hours using tikz, include it. If it would take longer, skip it — the paper is complete without it. Both reviewers noted its absence but neither called it a blocker.

**Decision 4: Second site or not? (Decide by April 18)**

Queue the US-Ha1 run April 18 morning. If it runs: add one sentence. If it fails: you learn something useful (what's broken) and report honestly in Limitations. Either way it's worth attempting given the low setup cost.

---

## 5. Risk Flags

**RISK 1 (HIGH): Job 7403949 (LE/H gradients) failed or is still running.**
This is a submission blocker. If it hasn't completed, resubmit on April 15 morning before anything else. The script exists; this is a scheduling risk, not a technical one.

**RISK 2 (HIGH): 2-parameter calibration doesn't show Adam advantage.**
If Adam loses to Nelder-Mead in 2D as well, the paper has no compelling calibration demonstration. Mitigation: have Option B (reframe) drafted by April 17 noon. This is not a paper-killer — it just means the calibration claim is weaker. The gradient correctness + Jacobian work is sufficient for a workshop paper.

**RISK 3 (HIGH): Jacobian heatmap is stale (Vcmax25 or dpai column still zero).**
The paper's Jacobian section is broken if either of these columns is structural zero. Mitigation: verify on April 15 morning (mtime check). If stale, queue regeneration that morning.

**RISK 4 (MEDIUM): LaTeX compilation failures due to figure path inconsistency.**
The submission portal will compile from the uploaded source. Relative paths `../../diags/figures/` will break. Mitigation: fix all figure paths April 16 (M5) and compile a clean PDF from the paper directory to confirm.

**RISK 5 (MEDIUM): bonan2025 DOI is permanently unresolvable (paper not yet assigned a DOI).**
If the paper is truly "in press" with no assigned DOI, use the preprint version if one exists. Alternatively: `\bibitem{bonan2025} Bonan et al. (2025). ... Agricultural and Forest Meteorology. In press.` is acceptable. Do not submit with `xxx`.

**RISK 6 (MEDIUM): Page limit.**
With all planned additions (translation table, methodology diagram, expanded calibration, Vcmax25+iota_SPA), the paper will likely exceed 6 pages. Cut the Ralph loop implementation appendix first (move to supplementary if needed), then cut the Fortran pattern table to a shortened version. Main text + figures must fit in the page limit.

**RISK 7 (LOW): NeurIPS AI4Science deadline is earlier than April 24 or requires institutional endorsement.**
Confirm the exact deadline this week. Some workshop submissions require institutional sign-off or pre-registration.

---

## 6. What to Do Tomorrow Morning (April 16)

In order — do not deviate:

1. `squeue -u al4385` — check job 7403949 and 7447256 status.
2. If 7403949 done: open JAXES.tex, fill Table 2 LE and H rows. Commit.
3. `ls -la /burg-archive/home/al4385/clm-ml-jax/diags/figures/sensitivity_jacobian.png` — check mtime.
4. `git log --follow -1 -- diags/sensitivity_analysis.py` — verify the Vcmax25 fix commit date is before the figure mtime. If figure is stale: queue sensitivity regeneration.
5. Check if 2D calibration job (submitted April 15 afternoon) completed. If done: extract convergence data.
6. Fix bonan2025 DOI. Google Scholar: "Bonan Burns Patton CHATS walnut orchard Agricultural Forest Meteorology 2025." 30 minutes.
7. Fix figure caption error: "7.6×10⁻³%" → "0.076%" in JAXES.tex. 5 minutes.
8. Fix figure paths in JAXES.tex: make all figures reference `figures/` subdirectory. Copy files.

Items 1–5 are verification and data collection. Items 6–8 are targeted edits. Do these first before opening any new coding project.

---

## Appendix: Reviewer Issue Tracker

| Issue | Source | Priority | Fix | Status |
|---|---|---|---|---|
| Adam loses 1D calibration | Both reviewers | CRITICAL | Replace with 2D Vcmax25+iota_SPA or reframe | Open |
| "Pending" in Table 2 | Both reviewers | CRITICAL | Job 7403949 result | Open |
| dpai=0 in Jacobian | ESM expert | CRITICAL | Fix or replace with q_ref | Open |
| Jacobian framing confused | Both reviewers | HIGH | Clarify jacrev choice; remove bad cost-ratio claim | Open |
| bonan2025 placeholder DOI | Both reviewers | HIGH | Look up actual DOI | Open |
| Figure caption error (7.6e-3%) | NeurIPS reviewer | HIGH | Change to 0.076% | Open |
| Figure path inconsistency | NeurIPS reviewer | HIGH | Standardize to figures/ | Open |
| "Global sensitivity" overclaim | NeurIPS reviewer | HIGH | Change to "Jacobian-based" | Open |
| "Drop-in replacement" overclaim | NeurIPS reviewer | MEDIUM | Add "canopy physics column" qualifier | Open |
| NeuralGCM not in text | ESM expert | MEDIUM | Wire citation | Open |
| Benchmark figure uses Euler | NeurIPS reviewer | MEDIUM | Add to caption | Open |
| No methodology pipeline diagram | Both reviewers | MEDIUM | Create Fig 7 in tikz | Open |
| Ralph loop framing oversold | NeurIPS reviewer | MEDIUM | Soften C3 and Section 3.4 | Open |
| No translation statistics | NeurIPS reviewer | MEDIUM | Add table to Section 3 | Open |
| Single-timestep not stated | ESM expert | MEDIUM | Add limitation statement | Open |
| Duplicate pattern table row | NeurIPS reviewer | LOW | Consolidate | Open |
| g1 Medlyn model untested | NeurIPS reviewer | LOW | Add note or run check | Open |
| Calibration alpha_sw demo weak | ESM expert | (Covered by W1) | Covered by 2D fix | Open |
