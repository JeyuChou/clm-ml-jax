"""
Generate publication figure for multi-parameter calibration experiment.

Loads diags/output/multipar_calibration_results.json and produces a 2-panel
figure at 7"x4" (NeurIPS column width):

  Panel A (left): Loss vs forward-equivalent evaluations
    - Adam/jax.grad (green solid)
    - L-BFGS-B/FD (orange dashed)
    - Nelder-Mead (blue dotted)
    - Vertical line at 2p forward-equiv (cost of first FD gradient step)

  Panel B (right): Gradient cost vs number of parameters p
    - AD cost (green flat line at T_ratio)
    - FD central (orange slope-2 line)
    - Shaded "AD faster" / "FD faster" regions
    - Empirical crossover point marked

Outputs:
  diags/figures/multipar_calibration.pdf
  diags/figures/multipar_calibration.png

Usage (from project root):
    python diags/plot_multipar_calibration.py
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

P          = res["p"]
T_ratio    = res["T_ratio"]
crossover  = res["crossover_p"]
T_fwd      = res["T_forward_s"]
T_bwd      = res["T_backward_s"]

adam     = res["adam"]
nm       = res["nelder_mead"]
lbfgsb   = res["lbfgsb_fd"]
timing   = res["timing_sweep"]

# ── Forward-equivalent axes ───────────────────────────────────────────────────
# Adam: each grad eval = T_ratio forward-equivalents
adam_x   = np.array(adam["fwd_equiv"],   dtype=float)
adam_y   = np.array(adam["loss_history"], dtype=float)

# NM: each function eval = 1 forward-equivalent
nm_x     = np.array(nm["nfev_history"],      dtype=float)
nm_y     = np.array(nm["loss_history"],      dtype=float)

# L-BFGS-B: each function eval = 1 forward-equivalent
lbfgsb_x = np.array(lbfgsb["nfev_history"], dtype=float)
lbfgsb_y = np.array(lbfgsb["loss_history"], dtype=float)

# Timing sweep
p_vals   = np.array(timing["p_values"], dtype=float)
T_ad_arr = np.array(timing["T_ad_s"],   dtype=float)
T_fd_arr = np.array(timing["T_fd_s"],   dtype=float)

# Normalize to T_forward = 1 for Panel B
T_ad_norm = T_ad_arr / T_fwd
T_fd_norm = T_fd_arr / T_fwd

# Fine p grid for smooth lines in Panel B
p_fine = np.logspace(np.log10(0.5), np.log10(60), 200)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 4))

# ── Panel A: convergence ──────────────────────────────────────────────────────
ax = axes[0]

final_err_adam   = adam["final_param_err"]
final_err_nm     = nm["final_param_err"]
final_err_lbfgsb = lbfgsb["final_param_err"]

if len(adam_x) > 0:
    ax.semilogy(adam_x, np.maximum(adam_y, 1e-16), color="#2ca02c", lw=1.8,
                label=f"Adam / jax.grad  ($\|\|\\theta-\\theta^*\|\|_2={final_err_adam:.3f}$)",
                zorder=3)
if len(lbfgsb_x) > 0:
    ax.semilogy(lbfgsb_x, np.maximum(lbfgsb_y, 1e-16), color="#ff7f0e",
                lw=1.8, linestyle="--",
                label=f"L-BFGS-B / FD  ($\|\|\\theta-\\theta^*\|\|_2={final_err_lbfgsb:.3f}$)",
                zorder=2)
if len(nm_x) > 0:
    ax.semilogy(nm_x, np.maximum(nm_y, 1e-16), color="#1f77b4",
                lw=1.8, linestyle=":",
                label=f"Nelder-Mead  ($\|\|\\theta-\\theta^*\|\|_2={final_err_nm:.3f}$)",
                zorder=1)

# Vertical line at cost of one FD gradient step (2p forward evals)
two_p = 2 * P
ax.axvline(two_p, color="gray", linestyle="-.", lw=1.2, alpha=0.8, zorder=0)
ax.text(two_p * 1.04, ax.get_ylim()[1] if ax.get_ylim()[1] > 1e-15 else 1.0,
        f"$2p={two_p}$ fwd-equiv\n(1 FD grad step)",
        fontsize=7, color="gray", va="top")

ax.set_xlabel("Forward-pass equivalents", fontsize=9)
ax.set_ylabel("Relative squared-error loss", fontsize=9)
ax.set_title(f"(a) Convergence — $p={P}$ parameters", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, which="both", alpha=0.25)
ax.tick_params(labelsize=8)

# Annotate T_ratio
ax.text(0.02, 0.04,
        f"$T_{{\\rm ratio}}={T_ratio:.1f}\\times$\ncrossover $p={crossover:.1f}$",
        transform=ax.transAxes, fontsize=7, color="#2ca02c",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

# ── Panel B: scaling ──────────────────────────────────────────────────────────
ax2 = axes[1]

ad_line = np.full_like(p_fine, T_ratio)   # constant = T_ratio * T_forward / T_forward
fd_line = 2.0 * p_fine                     # 2p * T_forward / T_forward = 2p

ax2.fill_between(p_fine, ad_line, fd_line,
                 where=fd_line >= ad_line,
                 color="#2ca02c", alpha=0.12, label="AD faster")
ax2.fill_between(p_fine, ad_line, fd_line,
                 where=fd_line < ad_line,
                 color="#ff7f0e", alpha=0.12, label="FD faster")

ax2.loglog(p_fine, ad_line, color="#2ca02c", lw=2.0, label="AD cost (constant)")
ax2.loglog(p_fine, fd_line, color="#ff7f0e", lw=2.0, linestyle="--",
           label="FD central ($2p$ fwd passes)")

# Empirical points from timing sweep
ax2.scatter(p_vals, T_ad_norm, color="#2ca02c", s=40, zorder=5)
ax2.scatter(p_vals, T_fd_norm, color="#ff7f0e", s=40, marker="s", zorder=5)

# Mark crossover
ax2.axvline(crossover, color="gray", linestyle="-.", lw=1.2, alpha=0.8)
ax2.text(crossover * 1.08, 1.2,
         f"crossover\n$p={crossover:.1f}$",
         fontsize=7, color="gray")

ax2.set_xlabel("Number of parameters $p$", fontsize=9)
ax2.set_ylabel("Cost (units of $T_{\\rm fwd}$)", fontsize=9)
ax2.set_title("(b) Gradient cost scaling", fontsize=9)
ax2.legend(fontsize=7, loc="upper left")
ax2.grid(True, which="both", alpha=0.25)
ax2.tick_params(labelsize=8)
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.set_xlim(0.9, 65)

fig.suptitle(
    f"CLM-ml-jax: AD vs FD vs Nelder-Mead — $p={P}$ atmospheric/physiological parameters\n"
    f"(CHATS7, May 1 2007;  $T_{{\\rm ratio}}={T_ratio:.1f}\\times$)",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

for ext in ("pdf", "png"):
    out = FIGURES_DIR / f"multipar_calibration.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

print("\n=== plot_multipar_calibration.py complete ===")
