"""
Experiment 3: Jacobian-based global sensitivity analysis.

Computes the full output-parameter Jacobian J = d[H, LE, GPP] / d[theta]
via reverse-mode autodiff (jax.jacrev), where theta is a vector of
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

# Ensure project root (parent of diags/) is on sys.path so 'diags' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
import multilayer_canopy.MLpftconMod as _pftcon_mod

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le, compute_h,
)

_orig_pftcon = _pftcon_mod.MLpftcon   # original MLpftcon instance

# ── Parameter names (for display) ─────────────────────────────────────────────
PARAM_NAMES  = ["Vcmax25", "T_air", "SW_rad", "q_ref\n(humidity)", "dpai\n(leaf area)"]
OUTPUT_NAMES = ["GPP", "H (canopy sum)", "LE (canopy sum)"]

_p   = grid.p
_n   = grid.ncan

# Build kwargs without atm2lnd_inst so we can pass it as a traced arg
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

# ── Multi-output forward function ────────────────────────────────────────────
def forward_multi(scales: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        scales: float64 array of shape (5,) — scale factors at baseline = 1.0
          0: alpha_vcmax25  — scale on vcmaxpft (passed as vcmaxpft_jax arg)
          1: alpha_tair     — scale on atm2lnd_inst.forc_t_downscaled_col
          2: alpha_sw       — scale on atm2lnd_inst.forc_solad & forc_solai
          3: alpha_qref     — scale on wateratm2lndbulk_inst.forc_q_downscaled_col
          4: alpha_dpai     — scale on mlcanopy_inst.dpai_profile

    Returns:
        jnp.ndarray of shape (3,): [GPP (umol CO2/m2/s), H_canopy (W/m2), LE_canopy (W/m2)]

    NOTE: Vcmax25 is passed via vcmaxpft_jax (bypasses JIT cache so gradient flows).
    Scaling vcmax25_profile/vcmax25_leaf directly in mlcanopy_inst does NOT work
    because CanopyNitrogenProfile (called inside _physics_step_fn) recomputes them
    from MLpftcon.vcmaxpft, overwriting the scaled values.

    T_air, SW_rad, q_ref are scaled via atm2lnd_inst/wateratm2lndbulk_inst because
    MLCanopyFluxes.__init__ copies forcing fields from those instances into
    mlcanopy_inst, overwriting any mlcanopy_inst forcing fields.

    dpai is scaled via mlcanopy_inst — it is NOT recomputed inside the physics step.
    It affects GPP via (a) nscale in CanopyNitrogenProfile and (b) the dpai weighting
    in compute_gpp/compute_le/compute_h.

    H and LE are computed from shleaf_leaf and lhleaf_leaf (leaf-level, weighted over
    canopy layers) — updated inside _physics_step_fn even in diff mode.
    """
    # Scale vcmax via vcmaxpft_jax — bypasses JIT cache so gradient flows correctly.
    vcmaxpft_jax = scales[0] * _orig_pftcon.vcmaxpft

    # Scale dpai via mlcanopy_inst (not recomputed inside physics step).
    modified_ml = mlcanopy_inst._replace(
        dpai_profile = scales[4] * mlcanopy_inst.dpai_profile,
    )
    # Scale forcing via atm2lnd_inst (actual source used by physics).
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_inst.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_inst.forc_solai_grc,
    )
    modified_watm = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col = scales[3] * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst=modified_ml,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=modified_watm,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )

    # Use leaf-level proxies (shleaf_leaf, lhleaf_leaf, agross_leaf) — all updated
    # by _physics_step_fn in diff mode.  _CanopyFluxesDiagnostics is skipped in
    # diff mode so gppveg_canopy, eflx_sh_tot, eflx_lh_tot are NOT updated.
    GPP = compute_gpp(inst, _p, _n)
    H   = compute_h(inst, _p, _n)
    LE  = compute_le(inst, _p, _n)

    return jnp.array([GPP, H, LE])


# ── Baseline outputs ─────────────────────────────────────────────────────────
print("\n=== Computing baseline outputs ===", flush=True)
scales0  = jnp.ones(5, dtype=jnp.float64)
baseline = forward_multi(scales0)
jax.block_until_ready(baseline)
print(f"  GPP = {float(baseline[0]):.3f} umol CO2/m2/s")
print(f"  H   = {float(baseline[1]):.3f} W/m2 (top canopy layer)")
print(f"  LE  = {float(baseline[2]):.6f} mol H2O/m2/s (top canopy layer)")

# ── Jacobian via jacfwd ───────────────────────────────────────────────────────
print("\n=== Computing Jacobian via jax.jacrev ===", flush=True)
jacrev_fn = jax.jit(jax.jacrev(forward_multi))

t0 = time.time()
J = jacrev_fn(scales0)      # shape: (3, 5)
jax.block_until_ready(J)
t_jacrev = time.time() - t0
print(f"  jacrev completed in {t_jacrev:.1f}s  (shape: {J.shape})")

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
n_outputs = 3
print(f"\n=== Runtime comparison ===")
print(f"  jacrev (autodiff, {n_outputs} outputs, {n_params} params): {t_jacrev:.1f}s")
print(f"  (jacrev cost ≈ {n_outputs} backward passes; FD would need {n_params}×2 = {n_params*2} forward evaluations)")
# Estimate single forward pass time for comparison
t0 = time.time()
_ = forward_multi(scales0)
jax.block_until_ready(_)
t_fwd = time.time() - t0
print(f"  Single forward pass: {t_fwd:.1f}s")
print(f"  FD equivalent cost estimate: ~{n_params * 2 * t_fwd:.1f}s")

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
    "(computed via jax.jacrev in a single reverse-mode pass)",
    fontsize=11,
)
fig.tight_layout()

out = FIGURES_DIR / "sensitivity_jacobian.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out}")
