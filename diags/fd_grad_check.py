"""
Experiment 2: Finite-difference gradient check.

Verifies that jax.grad produces accurate gradients through the full
CLM-ml-jax column by comparing against central finite differences for
two parameters with clear, non-trivial pathways to canopy GPP:

  - alpha_sw:     global scale factor on swskyb_forcing & swskyd_forcing
                  (solar radiation → PAR → electron transport → GPP)
  - alpha_vcmax25: global scale factor on vcmax25_profile & vcmax25_leaf
                  (Rubisco capacity → carboxylation-limited photosynthesis)

Output scalar: gppveg_canopy[p]  (total canopy GPP, umol CO2/m2/s)

Note: at high-light conditions the canopy may be light-limited (Aj < Ac),
in which case alpha_vcmax25 has a small but non-zero gradient (via the
co-limitation blending in the Farquhar model).  alpha_sw always has a
large gradient because it directly controls PAR → J.

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

# Ensure project root (parent of diags/) is on sys.path so 'diags' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_p = grid.p

# Build kwargs without atm2lnd_inst so we can pass it as a traced arg
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}


# ── GPP scalar forward functions ─────────────────────────────────────────────
# IMPORTANT: MLCanopyFluxes.__init__ overwrites swskyb_cur_forcing and
# tref_cur_forcing from atm2lnd_inst (lines 910-921 of MLCanopyFluxesMod.py).
# Gradients must flow through atm2lnd_inst, NOT mlcanopy_inst forcing fields.

def forward_gpp_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale beam+diffuse SW by alpha via atm2lnd_inst, return GPP."""
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col = alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = alpha * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


def forward_gpp_tref(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale air temperature by alpha via atm2lnd_inst, return GPP.

    Temperature affects photosynthesis via enzyme kinetics (Vcmax(T), Jmax(T)).
    """
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col = alpha * atm2lnd_inst.forc_t_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


# ── Print baseline values ────────────────────────────────────────────────────
print("\n=== Baseline outputs ===", flush=True)
baseline_gpp = float(forward_gpp_sw(jnp.float64(1.0)))
print(f"  GPP proxy (baseline) = {baseline_gpp:.4f} (agross_leaf weighted sum)", flush=True)

# ── Compute JAX gradients ─────────────────────────────────────────────────────
print("\n=== Computing JAX gradients ===", flush=True)

t0 = time.time()
grad_sw_jax = float(jax.jit(jax.grad(forward_gpp_sw))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_sw)   [JAX] = {grad_sw_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
grad_tref_jax = float(jax.jit(jax.grad(forward_gpp_tref))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_tref) [JAX] = {grad_tref_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

# ── Compute finite-difference gradients ───────────────────────────────────────
print("\n=== Computing finite-difference gradients ===", flush=True)
EPS = 1e-4

t0 = time.time()
f_plus  = float(forward_gpp_sw(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_sw(jnp.float64(1.0 - EPS)))
grad_sw_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_sw)   [FD]  = {grad_sw_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
f_plus  = float(forward_gpp_tref(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_tref(jnp.float64(1.0 - EPS)))
grad_tref_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_tref) [FD]  = {grad_tref_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n=== Gradient accuracy summary ===")
print(f"{'Parameter':<22}  {'JAX grad':>14}  {'FD grad':>14}  {'Rel error':>12}  {'Pass?':>6}")
print("-" * 76)

results = []
for name, jax_val, fd_val in [
    ("dGPP/d(alpha_sw)",   grad_sw_jax,   grad_sw_fd),
    ("dGPP/d(alpha_tref)", grad_tref_jax, grad_tref_fd),
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

fig.suptitle("CLM-ml-jax: Autodiff accuracy — dGPP/d(param) via jax.grad vs central FD\n(Exp 2)", fontsize=12)
fig.tight_layout()
out = FIGURES_DIR / "fd_grad_check.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")
