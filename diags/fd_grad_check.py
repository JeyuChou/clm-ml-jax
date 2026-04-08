"""
Experiment 2: Finite-difference gradient check.

Verifies that jax.grad produces accurate gradients through the full
CLM-ml-jax column by comparing against central finite differences for
two physiological parameters:
  - alpha_vcmax25: global scale factor on vcmax25_profile
  - alpha_dpai:    global scale factor on dpai_profile (canopy structure)

Usage (from project root):
    cd src && python ../diags/fd_grad_check.py

Output:
  - Console table: parameter, jax.grad value, FD value, relative error
  - diags/figures/fd_grad_check.png: bar chart of relative errors
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    forward_fn, mlcanopy_inst, grid, jax, jnp
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parameter wrappers ────────────────────────────────────────────────────────
def forward_vcmax_scale(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass with vcmax25_profile scaled by alpha."""
    modified = mlcanopy_inst._replace(
        vcmax25_profile = alpha * mlcanopy_inst.vcmax25_profile,
        vcmax25_leaf    = alpha * mlcanopy_inst.vcmax25_leaf,
    )
    return forward_fn(modified)


def forward_dpai_scale(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass with dpai_profile scaled by alpha."""
    modified = mlcanopy_inst._replace(
        dpai_profile = alpha * mlcanopy_inst.dpai_profile,
    )
    return forward_fn(modified)


# ── Compute JAX gradients ─────────────────────────────────────────────────────
print("\n=== Computing JAX gradients ===", flush=True)

t0 = time.time()
grad_vcmax_jax = float(jax.jit(jax.grad(forward_vcmax_scale))(jnp.float64(1.0)))
print(f"  grad(alpha_vcmax25) [JAX] = {grad_vcmax_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
grad_dpai_jax = float(jax.jit(jax.grad(forward_dpai_scale))(jnp.float64(1.0)))
print(f"  grad(alpha_dpai)    [JAX] = {grad_dpai_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

# ── Compute finite-difference gradients ───────────────────────────────────────
print("\n=== Computing finite-difference gradients ===", flush=True)
EPS = 1e-4

t0 = time.time()
f_plus  = float(forward_vcmax_scale(jnp.float64(1.0 + EPS)))
f_minus = float(forward_vcmax_scale(jnp.float64(1.0 - EPS)))
grad_vcmax_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  grad(alpha_vcmax25) [FD]  = {grad_vcmax_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
f_plus  = float(forward_dpai_scale(jnp.float64(1.0 + EPS)))
f_minus = float(forward_dpai_scale(jnp.float64(1.0 - EPS)))
grad_dpai_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  grad(alpha_dpai)    [FD]  = {grad_dpai_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n=== Gradient accuracy summary ===")
print(f"{'Parameter':<20}  {'JAX grad':>14}  {'FD grad':>14}  {'Rel error':>12}  {'Pass?':>6}")
print("-" * 72)

results = []
for name, jax_val, fd_val in [
    ("alpha_vcmax25", grad_vcmax_jax, grad_vcmax_fd),
    ("alpha_dpai",    grad_dpai_jax,  grad_dpai_fd),
]:
    rel_err = abs(jax_val - fd_val) / (abs(fd_val) + 1e-30)
    passed  = rel_err < 0.01  # 1% tolerance
    results.append((name, jax_val, fd_val, rel_err, passed))
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<18}  {jax_val:>14.4e}  {fd_val:>14.4e}  {rel_err:>12.2e}  {status:>6}")

all_pass = all(r[4] for r in results)
print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'} — "
      f"autodiff accuracy confirmed to {'<1%' if all_pass else '>1%'} relative error")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

names    = [r[0] for r in results]
jax_vals = np.abs([r[1] for r in results])
fd_vals  = np.abs([r[2] for r in results])
rel_errs = [r[3] for r in results]

x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, jax_vals, w, label="jax.grad", color="steelblue")
axes[0].bar(x + w/2, fd_vals,  w, label="FD (central)", color="coral", alpha=0.8)
axes[0].set_yscale("log")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names)
axes[0].set_ylabel("|gradient|")
axes[0].set_title("Gradient magnitudes: JAX vs FD")
axes[0].legend()

axes[1].bar(x, rel_errs, color=["green" if r[4] else "red" for r in results])
axes[1].axhline(0.01, color="k", linestyle="--", lw=1.2, label="1% threshold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_ylabel("Relative error")
axes[1].set_title("Gradient relative error (< 1% = pass)")
axes[1].legend()

fig.suptitle("CLM-ml-jax: Autodiff accuracy check (Exp 2)", fontsize=12)
fig.tight_layout()
out = FIGURES_DIR / "fd_grad_check.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")
