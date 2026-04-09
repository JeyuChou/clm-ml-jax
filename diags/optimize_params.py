"""
Gradient-based parameter optimization for CLM-ML-JAX.

Optimizes PFT-level physiological parameters (vcmaxpft, ...) by minimizing
a normalized RMSE loss between model-predicted and observed GPP (+ optionally LE).

Phase 1 (this implementation): vcmaxpft only, single timestep, synthetic observations.
- Generates "true" model output with CHATS7 parameters (vcmaxpft=125 μmol m-2 s-1)
- Starts from CLM default (vcmaxpft=57.7 μmol m-2 s-1)
- Verifies identifiability using jax.grad through CanopyNitrogenProfile

Phase 2 (future): multi-timestep optimization against real AmeriFlux observations.
See parameter_optimization_experiment.md for the full experiment design.

Differentiable path for vcmaxpft:
  alpha_vcmax * MLpftcon.vcmaxpft[pft]    (module-global mutation during trace)
  → CanopyNitrogenProfile: vcmax25top = MLpftcon.vcmaxpft[pft]  (JAX gather, no float())
  → vcmax25_leaf (per-layer JAX array)
  → LeafPhotosynthesis kernel (vcmax25_ic as JAX input)
  → agross_leaf → GPP loss

NOT YET differentiable (FD or re-run needed):
  iota_SPA: extracted with float(_iota_np[pft]) in LeafPhotosynthesis factory
  g1_MED: same issue

Usage (from project root):
    cd src && python ../diags/optimize_params.py [--synthetic] [--n-steps 200]

Output:
  - Console: loss curve, parameter trajectory
  - diags/figures/optimization_vcmax25.png: loss + parameter convergence plot
  - diags/figures/optimization_vcmax25_results.json: final parameters + metrics
"""
from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import multilayer_canopy.MLpftconMod as _pftcon_mod
from multilayer_canopy.MLclm_varpar import isun, isha

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}


# ── Forward function with injected vcmaxpft ───────────────────────────────────
def _run_with_vcmax_scale(alpha_vcmax: jnp.ndarray):
    """Run MLCanopyFluxes with vcmaxpft scaled by alpha_vcmax.

    Module-global mutation of MLpftconMod.MLpftcon injects the traced alpha
    into the JAX computation graph via MLpftcon.vcmaxpft[pft] (dynamic gather
    in CanopyNitrogenProfileMod.py — no np.asarray in the critical path).
    """
    original = _pftcon_mod.MLpftcon
    _pftcon_mod.MLpftcon = original._replace(
        vcmaxpft=alpha_vcmax * original.vcmaxpft
    )
    try:
        inst = MLCanopyFluxes(
            mlcanopy_inst=mlcanopy_inst,
            atm2lnd_inst=atm2lnd_inst,
            wateratm2lndbulk_inst=wateratm2lndbulk_inst,
            **_mlcf_kwargs_no_atm,
        )
        return inst
    finally:
        _pftcon_mod.MLpftcon = original


def forward_gpp(alpha_vcmax: jnp.ndarray) -> jnp.ndarray:
    """Return canopy GPP for given vcmaxpft scale factor."""
    return compute_gpp(_run_with_vcmax_scale(alpha_vcmax), _p, _ncan)


def forward_le(alpha_vcmax: jnp.ndarray) -> jnp.ndarray:
    """Return canopy LE proxy for given vcmaxpft scale factor."""
    return compute_le(_run_with_vcmax_scale(alpha_vcmax), _p, _ncan)


# ── Synthetic observations (identifiability test) ─────────────────────────────
def generate_synthetic_obs(vcmax_true: float = 125.0) -> dict:
    """Generate synthetic GPP+LE observations by running with 'true' vcmaxpft.

    Uses the CHATS7 site value (vcmaxpft[7] = 125) as the ground truth.
    The CLM default for PFT 7 is 57.7 μmol m-2 s-1.

    Returns dict with 'gpp_obs', 'le_obs', 'alpha_true' (scale relative to CLM default).
    """
    pft_default = float(_pftcon_mod.MLpftcon.vcmaxpft[7])
    alpha_true  = vcmax_true / pft_default
    print(f"\n=== Synthetic observations ===")
    print(f"  CLM default vcmaxpft[7] = {pft_default:.2f} μmol m-2 s-1")
    print(f"  True vcmaxpft[7]        = {vcmax_true:.2f} μmol m-2 s-1")
    print(f"  alpha_true              = {alpha_true:.4f}", flush=True)

    gpp_obs = float(forward_gpp(jnp.float64(alpha_true)))
    le_obs  = float(forward_le(jnp.float64(alpha_true)))
    print(f"  Synthetic GPP_obs = {gpp_obs:.4f}")
    print(f"  Synthetic LE_obs  = {le_obs:.4f}", flush=True)
    return {"gpp_obs": gpp_obs, "le_obs": le_obs, "alpha_true": alpha_true,
            "vcmax_true": vcmax_true, "vcmax_default": pft_default}


# ── Loss function ─────────────────────────────────────────────────────────────
def make_loss_fn(gpp_obs: float, le_obs: float,
                 w_gpp: float = 0.5, w_le: float = 0.5,
                 lam_reg: float = 0.05) -> callable:
    """Build the JAX-differentiable loss function.

    Loss = w_gpp * (GPP_model - GPP_obs)^2 / GPP_obs^2
         + w_le  * (LE_model  - LE_obs)^2  / LE_obs^2
         + lam_reg * log_alpha^2

    The normalization by obs^2 makes the loss scale-invariant.
    The regularization term pulls log(alpha) toward 0 (i.e., alpha toward 1).

    Parameters
    ----------
    gpp_obs : observed GPP (same units as compute_gpp output)
    le_obs  : observed LE (same units as compute_le output)
    w_gpp   : weight for GPP term
    w_le    : weight for LE term
    lam_reg : L2 regularization strength in log space
    """
    _gpp_obs = jnp.float64(gpp_obs)
    _le_obs  = jnp.float64(le_obs)

    def loss_fn(log_alpha: jnp.ndarray) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        gpp_model = forward_gpp(alpha)
        le_model  = forward_le(alpha)
        loss_gpp  = w_gpp * ((gpp_model - _gpp_obs) / (_gpp_obs + 1e-8)) ** 2
        loss_le   = w_le  * ((le_model  - _le_obs)  / (_le_obs  + 1e-8)) ** 2
        reg       = lam_reg * log_alpha ** 2
        return loss_gpp + loss_le + reg

    return loss_fn


# ── Adam optimizer (pure JAX) ─────────────────────────────────────────────────
def adam_init(params: jnp.ndarray, lr: float = 0.01,
              beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8) -> dict:
    return {
        "m": jnp.zeros_like(params),
        "v": jnp.zeros_like(params),
        "t": jnp.int32(0),
        "lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps,
    }


def adam_step(params: jnp.ndarray, grads: jnp.ndarray,
              state: dict) -> tuple[jnp.ndarray, dict]:
    t  = state["t"] + 1
    m  = state["beta1"] * state["m"] + (1.0 - state["beta1"]) * grads
    v  = state["beta2"] * state["v"] + (1.0 - state["beta2"]) * grads ** 2
    mh = m / (1.0 - state["beta1"] ** t)
    vh = v / (1.0 - state["beta2"] ** t)
    new_params = params - state["lr"] * mh / (jnp.sqrt(vh) + state["eps"])
    new_state = {**state, "m": m, "v": v, "t": t}
    return new_params, new_state


# ── Cosine annealing learning rate ──────────────────────────────────────��─────
def cosine_lr(step: int, lr_max: float = 0.01, lr_min: float = 0.001,
              period: int = 50) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * (step % period) / period))


# ── Main optimization loop ────────────────────────────────────────────────────
def run_optimization(
    loss_fn: callable,
    n_steps: int = 200,
    lr_max: float = 0.01,
    lr_min: float = 0.001,
    cosine_period: int = 50,
    patience: int = 20,
    tol_rel: float = 1e-4,
    alpha_true: float | None = None,
) -> dict:
    """Run Adam optimization of log_alpha.

    Parameters
    ----------
    loss_fn : callable, takes log_alpha (scalar jnp array), returns scalar loss
    n_steps : maximum optimizer steps
    lr_max : peak learning rate (cosine schedule)
    lr_min : minimum learning rate (cosine schedule)
    cosine_period : restart period (steps)
    patience : steps without improvement before early stopping
    tol_rel : relative loss improvement threshold for patience check
    alpha_true : ground truth alpha (for convergence monitoring in synthetic case)

    Returns
    -------
    dict with keys: alpha_final, loss_final, history, converged
    """
    print(f"\n=== Optimization loop ({n_steps} max steps) ===", flush=True)

    # Start from alpha=1.0 (CLM default)
    log_alpha = jnp.float64(0.0)
    opt_state = adam_init(log_alpha, lr=lr_max)

    val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Compile first step
    print("  Compiling JIT val_and_grad...", flush=True)
    t0 = time.time()
    _loss_init, _grad_init = val_and_grad(log_alpha)
    print(f"  Compilation done in {time.time()-t0:.1f}s. "
          f"Initial loss={float(_loss_init):.4f}, grad={float(_grad_init):.4e}", flush=True)

    history = {"step": [], "loss": [], "log_alpha": [], "alpha": [], "lr": []}
    best_loss = float("inf")
    no_improve_count = 0
    converged = False

    for step in range(n_steps):
        # Cosine annealing LR
        lr = cosine_lr(step, lr_max=lr_max, lr_min=lr_min, period=cosine_period)
        opt_state["lr"] = lr

        t0 = time.time()
        loss_val, grad_val = val_and_grad(log_alpha)
        elapsed = time.time() - t0

        log_alpha, opt_state = adam_step(log_alpha, grad_val, opt_state)

        loss_f   = float(loss_val)
        alpha_f  = float(jnp.exp(log_alpha))
        log_a_f  = float(log_alpha)

        history["step"].append(step)
        history["loss"].append(loss_f)
        history["log_alpha"].append(log_a_f)
        history["alpha"].append(alpha_f)
        history["lr"].append(lr)

        # Patience check
        if loss_f < best_loss * (1.0 - tol_rel):
            best_loss = loss_f
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Logging
        if step % 10 == 0 or step < 5:
            vcmax_current = alpha_f * float(_pftcon_mod.MLpftcon.vcmaxpft[7])
            true_str = ""
            if alpha_true is not None:
                vcmax_true = alpha_true * float(_pftcon_mod.MLpftcon.vcmaxpft[7])
                err_pct = abs(vcmax_current - vcmax_true) / vcmax_true * 100
                true_str = f"  err={err_pct:.1f}%"
            print(f"  step {step:4d}  loss={loss_f:.5f}  alpha={alpha_f:.4f}  "
                  f"vcmax25={vcmax_current:.2f}  lr={lr:.4f}{true_str}  ({elapsed:.1f}s/step)",
                  flush=True)

        # Early stopping
        if no_improve_count >= patience:
            print(f"  Early stopping at step {step}: no improvement for {patience} steps",
                  flush=True)
            converged = True
            break

        # Gradient explosion check
        if not np.isfinite(float(grad_val)) or abs(float(grad_val)) > 1e6:
            print(f"  ABORT: gradient explosion at step {step}: grad={float(grad_val):.2e}",
                  flush=True)
            break

    final_alpha    = float(jnp.exp(log_alpha))
    final_vcmax25  = final_alpha * float(_pftcon_mod.MLpftcon.vcmaxpft[7])
    default_vcmax25 = float(_pftcon_mod.MLpftcon.vcmaxpft[7])

    print(f"\n=== Optimization complete ===")
    print(f"  CLM default vcmaxpft[7] = {default_vcmax25:.2f} μmol m-2 s-1")
    if alpha_true is not None:
        true_vcmax = alpha_true * default_vcmax25
        print(f"  True vcmaxpft[7]        = {true_vcmax:.2f} μmol m-2 s-1")
    print(f"  Optimized vcmaxpft[7]   = {final_vcmax25:.2f} μmol m-2 s-1")
    print(f"  Final loss              = {history['loss'][-1]:.5f}")

    return {
        "alpha_final": final_alpha,
        "vcmax25_final": final_vcmax25,
        "vcmax25_default": default_vcmax25,
        "loss_final": history["loss"][-1],
        "history": history,
        "converged": converged,
        "n_steps_run": len(history["step"]),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: dict, alpha_true: float | None = None, out_path=None):
    """Plot loss curve and parameter convergence."""
    h  = results["history"]
    default_vcmax = results["vcmax25_default"]
    opt_vcmax     = results["vcmax25_final"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss curve
    axes[0].semilogy(h["step"], h["loss"], "b-", linewidth=1.5)
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].grid(True, alpha=0.3)

    # vcmax25 trajectory
    vcmax_traj = [a * default_vcmax for a in h["alpha"]]
    axes[1].plot(h["step"], vcmax_traj, "b-", linewidth=1.5, label="Optimized")
    axes[1].axhline(default_vcmax, color="gray", linestyle="--", label=f"Default ({default_vcmax:.1f})")
    if alpha_true is not None:
        true_vcmax = alpha_true * default_vcmax
        axes[1].axhline(true_vcmax, color="red", linestyle="--", label=f"True ({true_vcmax:.1f})")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("vcmax25  (μmol m⁻² s⁻¹)")
    axes[1].set_title("vcmax25 convergence")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[2].plot(h["step"], h["lr"], "g-", linewidth=1.5)
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Cosine LR schedule")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"CLM-ML-JAX vcmax25 optimization — CHATS7 synthetic case\n"
        f"Recovered: {opt_vcmax:.2f} μmol m⁻² s⁻¹  "
        f"(default={default_vcmax:.1f}, "
        f"true={alpha_true*default_vcmax:.1f if alpha_true else 'N/A'})",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = FIGURES_DIR / "optimization_vcmax25.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CLM-ML-JAX parameter optimization")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic observations (identifiability test)")
    parser.add_argument("--vcmax-true", type=float, default=125.0,
                        help="True vcmaxpft value for synthetic case (default: 125.0)")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Maximum optimizer steps (default: 200)")
    parser.add_argument("--lr-max", type=float, default=0.01,
                        help="Peak learning rate (default: 0.01)")
    parser.add_argument("--w-gpp", type=float, default=0.5,
                        help="GPP loss weight (default: 0.5)")
    parser.add_argument("--w-le", type=float, default=0.5,
                        help="LE loss weight (default: 0.5)")
    parser.add_argument("--lam-reg", type=float, default=0.05,
                        help="L2 regularization strength (default: 0.05)")
    args = parser.parse_args()

    # ── Generate observations ─────────────────────────────────────────────────
    if args.synthetic:
        obs_info = generate_synthetic_obs(vcmax_true=args.vcmax_true)
    else:
        raise NotImplementedError(
            "Real observations require --obs-csv path. "
            "See diags/expt_load_obs.py for the observation loader."
        )

    # ── Build loss function ───────────────────────────────────────────────────
    loss_fn = make_loss_fn(
        gpp_obs=obs_info["gpp_obs"],
        le_obs=obs_info["le_obs"],
        w_gpp=args.w_gpp,
        w_le=args.w_le,
        lam_reg=args.lam_reg,
    )

    # Verify gradient is finite at starting point before running optimization
    print("\n=== Pre-optimization gradient check ===", flush=True)
    t0 = time.time()
    _loss0, _grad0 = jax.value_and_grad(loss_fn)(jnp.float64(0.0))
    print(f"  loss at alpha=1.0: {float(_loss0):.5f}", flush=True)
    print(f"  grad at alpha=1.0: {float(_grad0):.4e}  ({time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(float(_grad0)):
        print("  ERROR: gradient is not finite at starting point. "
              "Check bracket_ok fix in MLLeafPhotosynthesisMod.py.", flush=True)
        return

    # ── Run optimization ──────────────────────────────────────────────────────
    results = run_optimization(
        loss_fn=loss_fn,
        n_steps=args.n_steps,
        lr_max=args.lr_max,
        lr_min=args.lr_max / 10.0,
        alpha_true=obs_info.get("alpha_true"),
    )
    results["obs_info"] = obs_info

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = FIGURES_DIR / "optimization_vcmax25_results.json"
    # Convert jnp arrays to Python scalars for JSON serialization
    save_results = {k: (float(v) if hasattr(v, "__float__") else v)
                    for k, v in results.items()
                    if k != "history"}
    save_results["history"] = {k: [float(x) for x in v]
                                for k, v in results["history"].items()}
    with open(out_json, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(results, alpha_true=obs_info.get("alpha_true"))


if __name__ == "__main__":
    main()
