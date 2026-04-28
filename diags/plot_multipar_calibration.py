"""
Generate publication figure for multi-parameter calibration experiment.

Loads diags/output/multipar_calibration_results.json and produces a 2-panel
figure at 7"x4" (NeurIPS column width):

  Panel A (left): Loss vs forward-equivalent evaluations — convergence
    - Adam + jax.grad (green solid)
    - L-BFGS-B + jax.grad (purple dashed-dot)
    - L-BFGS-B + FD (orange dashed)
    - Nelder-Mead (blue dotted)
    - Vertical line at 2p forward-equiv (cost of one FD gradient step)

  Panel B (right): Gradient cost vs number of parameters p
    - AD cost (green flat at T_ratio)
    - FD central (orange slope-2)
    - "AD faster" / "FD faster" shaded regions

Outputs:
  diags/figures/multipar_calibration.pdf
  diags/figures/multipar_calibration.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).parent / "output" / "multipar_calibration_results.json"
if not DATA_PATH.exists():
    print(f"ERROR: {DATA_PATH} not found. Run diags/multipar_calibration.py first.")
    sys.exit(1)

with open(DATA_PATH) as f:
    res = json.load(f)

P         = res["p"]
T_ratio   = res["T_ratio"]
crossover = res["crossover_p"]
T_fwd     = res["T_forward_s"]
T_bwd     = res["T_backward_s"]

adam     = res["adam"]
lbfgsb_ad = res.get("lbfgsb_ad", None)
lbfgsb_fd = res["lbfgsb_fd"]
nm        = res["nelder_mead"]
timing    = res["timing_sweep"]

# ── Forward-equivalent axes ───────────────────────────────────────────────────
# Adam: each grad eval = T_ratio forward-equivalents
adam_x = np.array(adam["fwd_equiv"], dtype=float)
adam_y = np.array(adam["loss_history"], dtype=float)

# L-BFGS-B/AD: each grad+loss eval = T_ratio forward-equivalents
if lbfgsb_ad is not None:
    lbfgsb_ad_x = np.array(lbfgsb_ad["fwd_equiv"], dtype=float)
    lbfgsb_ad_y = np.array(lbfgsb_ad["loss_history"], dtype=float)

# L-BFGS-B/FD: each function eval = 1 forward-equivalent
lbfgsb_fd_x = np.array(lbfgsb_fd["nfev_history"], dtype=float)
lbfgsb_fd_y = np.array(lbfgsb_fd["loss_history"], dtype=float)

# NM: each function eval = 1 forward-equivalent
nm_x = np.array(nm["nfev_history"], dtype=float)
nm_y = np.array(nm["loss_history"], dtype=float)

# Timing sweep for Panel B
p_vals   = np.array(timing["p_values"], dtype=float)
T_ad_arr = np.array(timing["T_ad_s"], dtype=float)
T_fd_arr = np.array(timing["T_fd_s"], dtype=float)
T_ad_norm = T_ad_arr / T_fwd
T_fd_norm = T_fd_arr / T_fwd
p_fine = np.logspace(np.log10(0.5), np.log10(60), 200)

# ── Colors and styles ─────────────────────────────────────────────────────────
C_ADAM    = "#2ca02c"   # green  — Adam/AD
C_LBFGSB  = "#9467bd"  # purple — L-BFGS-B/AD
C_FD      = "#ff7f0e"   # orange — L-BFGS-B/FD
C_NM      = "#1f77b4"   # blue   — Nelder-Mead

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 4))

# ── Panel A: convergence ──────────────────────────────────────────────────────
ax = axes[0]

final_err_adam    = adam["final_param_err"]
final_err_lbad    = lbfgsb_ad["final_param_err"] if lbfgsb_ad else None
final_err_lbfd    = lbfgsb_fd["final_param_err"]
final_err_nm      = nm["final_param_err"]

if len(adam_x) > 0:
    ax.semilogy(adam_x, np.maximum(adam_y, 1e-16),
                color=C_ADAM, lw=1.8,
                label=f"Adam + jax.grad  ($\|\|\\theta-\\theta^*\|\|={final_err_adam:.3f}$)",
                zorder=4)

if lbfgsb_ad is not None and len(lbfgsb_ad_x) > 0:
    ax.semilogy(lbfgsb_ad_x, np.maximum(lbfgsb_ad_y, 1e-16),
                color=C_LBFGSB, lw=1.8, linestyle="-.",
                label=f"L-BFGS-B + jax.grad  ($\|\|\\theta-\\theta^*\|\|={final_err_lbad:.3f}$)",
                zorder=5)

if len(lbfgsb_fd_x) > 0:
    ax.semilogy(lbfgsb_fd_x, np.maximum(lbfgsb_fd_y, 1e-16),
                color=C_FD, lw=1.8, linestyle="--",
                label=f"L-BFGS-B + FD  ($\|\|\\theta-\\theta^*\|\|={final_err_lbfd:.3f}$)",
                zorder=2)

if len(nm_x) > 0:
    ax.semilogy(nm_x, np.maximum(nm_y, 1e-16),
                color=C_NM, lw=1.8, linestyle=":",
                label=f"Nelder-Mead  ($\|\|\\theta-\\theta^*\|\|={final_err_nm:.3f}$)",
                zorder=1)

# Vertical line at cost of one FD gradient step (2p forward evals, central diff)
two_p = 2 * P
ax.axvline(two_p, color="gray", linestyle="-.", lw=1.0, alpha=0.7, zorder=0)
ax.text(two_p * 1.04, 1.0,
        f"$2p={two_p}$\nfwd-equiv\n(1 FD grad)",
        fontsize=6.5, color="gray", va="top")

ax.set_xlabel("Forward-pass equivalents", fontsize=9)
ax.set_ylabel("Combined loss (GPP + H + LE)", fontsize=9)
ax.set_title(f"(a) Convergence — $p={P}$ parameters (all active)", fontsize=9)
ax.legend(fontsize=6.5, loc="upper right")
ax.grid(True, which="both", alpha=0.25)
ax.tick_params(labelsize=8)

ax.text(0.02, 0.04,
        f"$T_{{\\rm ratio}}={T_ratio:.1f}\\times$\ncrossover $p={crossover:.1f}$",
        transform=ax.transAxes, fontsize=7, color=C_ADAM,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

# ── Panel B: scaling ──────────────────────────────────────────────────────────
ax2 = axes[1]

ad_line = np.full_like(p_fine, T_ratio)
fd_line = 2.0 * p_fine

ax2.fill_between(p_fine, ad_line, fd_line,
                 where=fd_line >= ad_line,
                 color=C_ADAM, alpha=0.12, label="AD faster")
ax2.fill_between(p_fine, ad_line, fd_line,
                 where=fd_line < ad_line,
                 color=C_FD, alpha=0.12, label="FD faster")

ax2.loglog(p_fine, ad_line, color=C_ADAM, lw=2.0, label="AD cost (constant)")
ax2.loglog(p_fine, fd_line, color=C_FD,   lw=2.0, linestyle="--",
           label="FD central ($2p$ fwd passes)")

ax2.scatter(p_vals, T_ad_norm, color=C_ADAM, s=40, zorder=5)
ax2.scatter(p_vals, T_fd_norm, color=C_FD,   s=40, marker="s", zorder=5)

# Mark crossover
ax2.axvline(crossover, color="gray", linestyle="-.", lw=1.0, alpha=0.7)
ax2.text(crossover * 1.08, 1.2,
         f"crossover\n$p={crossover:.1f}$",
         fontsize=7, color="gray")

# Mark p=10 experiment point
ax2.axvline(P, color=C_ADAM, linestyle=":", lw=1.0, alpha=0.5)
ax2.text(P * 1.06, ax2.get_ylim()[1] if hasattr(ax2, '_ylim') else T_ratio * 3,
         f"$p={P}$\nexperiment",
         fontsize=6.5, color=C_ADAM, va="top")

ax2.set_xlabel("Number of parameters $p$", fontsize=9)
ax2.set_ylabel("Cost (units of $T_{\\rm fwd}$)", fontsize=9)
ax2.set_title("(b) Gradient cost scaling", fontsize=9)
ax2.legend(fontsize=7, loc="upper left")
ax2.grid(True, which="both", alpha=0.25)
ax2.tick_params(labelsize=8)
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.set_xlim(0.9, 65)

fig.suptitle(
    f"CLM-ml-jax: gradient-based vs gradient-free calibration — $p={P}$ parameters (all active)\n"
    f"Loss: normalized MSE over GPP+H+LE  ·  "
    f"$T_{{\\rm ratio}}={T_ratio:.1f}\\times$  ·  CHATS7, May 1 2007",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

for ext in ("pdf", "png"):
    out = FIGURES_DIR / f"multipar_calibration.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Method':<25}  {'Final loss':>12}  {'||err||_2':>10}  {'Converged':>10}")
print("-" * 65)
print(f"{'Adam + jax.grad':<25}  {adam['final_loss']:>12.4e}  "
      f"{adam['final_param_err']:>10.4f}  {'—':>10}")
if lbfgsb_ad:
    print(f"{'L-BFGS-B + jax.grad':<25}  {lbfgsb_ad['final_loss']:>12.4e}  "
          f"{lbfgsb_ad['final_param_err']:>10.4f}  "
          f"{'yes' if lbfgsb_ad['converged'] else 'no':>10}")
print(f"{'L-BFGS-B + FD':<25}  {lbfgsb_fd['final_loss']:>12.4e}  "
      f"{lbfgsb_fd['final_param_err']:>10.4f}  "
      f"{'yes' if lbfgsb_fd['converged'] else 'no':>10}")
print(f"{'Nelder-Mead':<25}  {nm['final_loss']:>12.4e}  "
      f"{nm['final_param_err']:>10.4f}  "
      f"{'yes' if nm['converged'] else 'no':>10}")

print("\n=== plot_multipar_calibration.py complete ===")
