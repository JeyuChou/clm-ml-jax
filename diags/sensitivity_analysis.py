"""
Experiment 3: Jacobian-based global sensitivity analysis.

Computes the full output-parameter Jacobian J = d[H, LE, GPP] / d[theta]
via forward-mode autodiff (jax.jacfwd), where theta is a vector of
five physiological and forcing scale factors:
  0: alpha_vcmax25  — scale on vcmax25_profile & vcmax25_leaf (Vcmax25)
  1: alpha_tref     — scale on tref_forcing (air temperature)
  2: alpha_sw       — scale on swskyb_forcing & swskyd_forcing (solar radiation)
  3: alpha_qref     — scale on qref_forcing (specific humidity / VPD)
  4: alpha_dpai     — scale on dpai_profile (canopy leaf area)

Outputs: [H_sum, LE_sum, GPP_canopy] at the single CHATS7 patch.

Usage (from project root):
    cd src && python ../diags/sensitivity_analysis.py

Output:
  - Console: Jacobian values and runtime comparison with FD cost estimate
  - diags/figures/sensitivity_jacobian.png: normalised heatmap (paper figure)
  - diags/figures/sensitivity_jacobian.csv: raw Jacobian values
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    forward_fn, mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes
)

# ── Parameter names (for display) ─────────────────────────────────────────────
PARAM_NAMES  = ["Vcmax25", "T_air", "SW_rad", "q_ref\n(humidity)", "dpai\n(leaf area)"]
OUTPUT_NAMES = ["H (sensible)", "LE (latent)", "GPP"]

_p   = grid.p
_n   = grid.ncan

# ── Multi-output forward function ────────────────────────────────────────────
def forward_multi(scales: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        scales: float64 array of shape (5,) — scale factors at baseline = 1.0

    Returns:
        jnp.ndarray of shape (3,): [H_sum (W/m2), LE_sum (mol/m2/s), GPP (umol/m2/s)]
    """
    modified = mlcanopy_inst._replace(
        vcmax25_profile  = scales[0] * mlcanopy_inst.vcmax25_profile,
        vcmax25_leaf     = scales[0] * mlcanopy_inst.vcmax25_leaf,
        tref_forcing     = scales[1] * mlcanopy_inst.tref_forcing,
        tref_bef_forcing = scales[1] * mlcanopy_inst.tref_bef_forcing,
        tref_cur_forcing = scales[1] * mlcanopy_inst.tref_cur_forcing,
        tref_next_forcing= scales[1] * mlcanopy_inst.tref_next_forcing,
        swskyb_forcing   = scales[2] * mlcanopy_inst.swskyb_forcing,
        swskyb_bef_forcing=scales[2] * mlcanopy_inst.swskyb_bef_forcing,
        swskyb_cur_forcing=scales[2] * mlcanopy_inst.swskyb_cur_forcing,
        swskyb_next_forcing=scales[2] * mlcanopy_inst.swskyb_next_forcing,
        swskyd_forcing   = scales[2] * mlcanopy_inst.swskyd_forcing,
        swskyd_bef_forcing=scales[2] * mlcanopy_inst.swskyd_bef_forcing,
        swskyd_cur_forcing=scales[2] * mlcanopy_inst.swskyd_cur_forcing,
        swskyd_next_forcing=scales[2] * mlcanopy_inst.swskyd_next_forcing,
        qref_forcing     = scales[3] * mlcanopy_inst.qref_forcing,
        qref_bef_forcing = scales[3] * mlcanopy_inst.qref_bef_forcing,
        qref_cur_forcing = scales[3] * mlcanopy_inst.qref_cur_forcing,
        qref_next_forcing= scales[3] * mlcanopy_inst.qref_next_forcing,
        dpai_profile     = scales[4] * mlcanopy_inst.dpai_profile,
    )
    inst = MLCanopyFluxes(mlcanopy_inst=modified, **_mlcf_kwargs)

    H   = jnp.sum(inst.shair_profile[_p, 1:_n + 1])       # W/m2 (summed over layers)
    LE  = jnp.sum(inst.etair_profile[_p, 1:_n + 1])        # mol H2O/m2/s (sum over layers)
    GPP = inst.gppveg_canopy[_p]                            # umol CO2/m2/s

    return jnp.array([H, LE, GPP])


# ── Baseline outputs ─────────────────────────────────────────────────────────
print("\n=== Computing baseline outputs ===", flush=True)
scales0  = jnp.ones(5, dtype=jnp.float64)
baseline = forward_multi(scales0)
jax.block_until_ready(baseline)
print(f"  H   = {float(baseline[0]):.3f} W/m2 (sum over {_n} layers)")
print(f"  LE  = {float(baseline[1]):.6f} mol H2O/m2/s (sum over {_n} layers)")
print(f"  GPP = {float(baseline[2]):.3f} umol CO2/m2/s")

# ── Jacobian via jacfwd ───────────────────────────────────────────────────────
print("\n=== Computing Jacobian via jax.jacfwd ===", flush=True)
jacfwd_fn = jax.jit(jax.jacfwd(forward_multi))

t0 = time.time()
J = jacfwd_fn(scales0)      # shape: (3, 5)
jax.block_until_ready(J)
t_jacfwd = time.time() - t0
print(f"  jacfwd completed in {t_jacfwd:.1f}s  (shape: {J.shape})")

J_np = np.array(J)
print("\n  Raw Jacobian J[i,j] = d(output_i)/d(scale_j):")
print(f"  {'':15s}  " + "  ".join(f"{p:>12s}" for p in PARAM_NAMES))
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_np[i,j]:>12.4e}" for j in range(5))
    print(f"  {oname:15s}  {row}")

# ── Normalised sensitivity ────────────────────────────────────────────────────
# Normalize: J_norm[i,j] = J[i,j] / std(output_i)  so rows are comparable
# Using output baseline values as approximate std (1% perturbation interpretation)
output_scale = np.abs(J_np).max(axis=1, keepdims=True) + 1e-30
J_norm = J_np / output_scale

# ── Runtime comparison ────────────────────────────────────────────────────────
n_params = 5
print(f"\n=== Runtime comparison ===")
print(f"  jacfwd (autodiff, {n_params} params, 3 outputs): {t_jacfwd:.1f}s")
print(f"  FD equivalent ({n_params} × 2 forward passes):   ~{n_params * 2 * t_jacfwd / n_params:.1f}s")
print(f"  (jacfwd cost ≈ {n_params} forward-mode passes; FD would require {n_params} × 2 = {n_params*2} evaluations)")

# ── Save CSV ──────────────────────────────────────────────────────────────────
import csv
csv_path = FIGURES_DIR / "sensitivity_jacobian.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["output"] + PARAM_NAMES)
    for i, oname in enumerate(OUTPUT_NAMES):
        writer.writerow([oname] + list(J_np[i]))
print(f"\nRaw Jacobian saved: {csv_path}")

# ── Figure: normalised heatmap ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: absolute Jacobian (log scale magnitude)
log_J = np.log10(np.abs(J_np) + 1e-30)
im0 = axes[0].imshow(log_J, aspect="auto", cmap="RdBu_r",
                      vmin=log_J.min(), vmax=log_J.max())
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(PARAM_NAMES, fontsize=9)
axes[0].set_yticks(range(3))
axes[0].set_yticklabels(OUTPUT_NAMES, fontsize=9)
for i in range(3):
    for j in range(5):
        axes[0].text(j, i, f"{J_np[i,j]:.2e}", ha="center", va="center",
                     fontsize=7, color="k")
axes[0].set_title("log₁₀|∂output/∂scale|", fontsize=11)
plt.colorbar(im0, ax=axes[0], label="log₁₀|J|")

# Right: row-normalised (relative sensitivity within each output)
divnorm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im1 = axes[1].imshow(J_norm, aspect="auto", cmap="RdBu_r", norm=divnorm)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(PARAM_NAMES, fontsize=9)
axes[1].set_yticks(range(3))
axes[1].set_yticklabels(OUTPUT_NAMES, fontsize=9)
for i in range(3):
    for j in range(5):
        axes[1].text(j, i, f"{J_norm[i,j]:+.2f}", ha="center", va="center",
                     fontsize=9, color="k")
axes[1].set_title("Normalised sensitivity\n(row max = ±1)", fontsize=11)
plt.colorbar(im1, ax=axes[1], label="Normalised ∂output/∂scale")

fig.suptitle(
    "CLM-ml-jax: Jacobian sensitivity analysis — CHATS7, May 1 2007, t=1\n"
    "(computed via jax.jacfwd in a single forward-mode pass)",
    fontsize=11,
)
fig.tight_layout()

out = FIGURES_DIR / "sensitivity_jacobian.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out}")
