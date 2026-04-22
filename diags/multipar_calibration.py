"""
Multi-parameter calibration experiment (p=10): AD vs FD vs Nelder-Mead.

Demonstrates the key advantage of reverse-mode AD (jax.grad):
  - T_AD = O(1 backward pass), independent of p
  - T_FD = O(2p forward passes) for central differences
  - Crossover: AD wins when p > T_ratio / 2

Parameter set (p=10 scale factors, all initialized to 1.0 at truth):
  0  alpha_sw    — shortwave radiation (direct + diffuse)
  1  alpha_tref  — air temperature
  2  alpha_vcmax — global Vcmax25 (scales entire vcmaxpft array)
  3  alpha_iota  — WUE efficiency iota_SPA (scales entire array)
  4  alpha_q     — specific humidity
  5  alpha_pbot  — atmospheric pressure
  6  alpha_lwrad — longwave radiation
  7  alpha_u     — wind u-component
  8  alpha_v     — wind v-component
  9  alpha_pco2  — CO2 partial pressure

All scale factors in linear space (theta = alpha, truth = ones(10)).

Outputs:
  diags/output/multipar_calibration_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _d in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
print("=== multipar_calibration.py: loading model ===", flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp,
)
import multilayer_canopy.MLpftconMod             as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon

P = 10   # number of parameters

# ── pftcon injection helpers ──────────────────────────────────────────────────

def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── p=10 forward function ─────────────────────────────────────────────────────

def forward_gpp_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """p=10 differentiable forward pass. theta shape: (10,). Returns scalar GPP.

    theta[0] alpha_sw    — scale on shortwave (direct + diffuse)
    theta[1] alpha_tref  — scale on air temperature
    theta[2] alpha_vcmax — global scale on vcmaxpft array
    theta[3] alpha_iota  — global scale on iota_SPA array
    theta[4] alpha_q     — scale on specific humidity
    theta[5] alpha_pbot  — scale on atmospheric pressure
    theta[6] alpha_lwrad — scale on longwave radiation
    theta[7] alpha_u     — scale on wind u-component
    theta[8] alpha_v     — scale on wind v-component
    theta[9] alpha_pco2  — scale on CO2 partial pressure
    """
    # Build modified atm2lnd_inst
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col = theta[0] * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = theta[0] * atm2lnd_inst.forc_solai_grc,
        forc_t_downscaled_col     = theta[1] * atm2lnd_inst.forc_t_downscaled_col,
        forc_lwrad_downscaled_col = theta[6] * atm2lnd_inst.forc_lwrad_downscaled_col,
        forc_pbot_downscaled_col  = theta[5] * atm2lnd_inst.forc_pbot_downscaled_col,
        forc_u_grc                = theta[7] * atm2lnd_inst.forc_u_grc,
        forc_v_grc                = theta[8] * atm2lnd_inst.forc_v_grc,
        forc_pco2_grc             = theta[9] * atm2lnd_inst.forc_pco2_grc,
    )

    # Humidity scale via wateratm2lndbulk_inst
    modified_wat = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col = theta[4] * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )

    # iota_SPA: module-global mutation (JAX traces through jnp.asarray)
    _set_pftcon(_orig_pftcon._replace(
        iota_SPA=theta[3] * _orig_pftcon.iota_SPA,
    ))

    # vcmaxpft: explicit JAX arg to bypass JIT cache
    vcmaxpft_jax = theta[2] * _orig_pftcon.vcmaxpft

    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=modified_wat,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()
    return compute_gpp(inst, _p, _ncan)


# ── Generate synthetic target ─────────────────────────────────────────────────
theta_star = jnp.ones(P, dtype=jnp.float64)

print("\n=== Generating synthetic target (theta_star = ones) ===", flush=True)
t0 = time.time()
GPP_target = forward_gpp_theta(theta_star)
jax.block_until_ready(GPP_target)
print(f"  GPP_target = {float(GPP_target):.6f}  ({time.time()-t0:.2f}s)", flush=True)

if float(GPP_target) == 0.0:
    print("  WARNING: GPP_target is zero — likely nighttime step. Gradients will be uninformative.", flush=True)


def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
    """Relative squared error loss."""
    gpp = forward_gpp_theta(theta)
    return ((gpp - GPP_target) / (jnp.abs(GPP_target) + 1e-6)) ** 2


# ── Timing: single forward pass ───────────────────────────────────────────────
print("\n=== Timing: single JIT-compiled forward pass ===", flush=True)
_loss_jit = jax.jit(loss_fn)

# Warmup (includes compilation)
print("  Warmup (JIT compile)...", flush=True)
t_warmup = time.time()
_ = jax.block_until_ready(_loss_jit(theta_star))
t_warmup = time.time() - t_warmup
print(f"  JIT compile + first eval: {t_warmup:.2f}s", flush=True)

# Measure forward pass (cached)
N_TIMING = 5
t_fwd_runs = []
for _ in range(N_TIMING):
    t0 = time.time()
    _ = jax.block_until_ready(_loss_jit(theta_star))
    t_fwd_runs.append(time.time() - t0)
T_forward_s = float(np.median(t_fwd_runs))
print(f"  T_forward (median of {N_TIMING} runs): {T_forward_s:.4f}s", flush=True)


# ── Timing: single backward pass ─────────────────────────────────────────────
print("\n=== Timing: JIT-compiled backward pass ===", flush=True)
_loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

print("  Warmup backward (JIT compile)...", flush=True)
t_warmup_bwd = time.time()
_ = jax.block_until_ready(_loss_and_grad(theta_star))
t_warmup_bwd = time.time() - t_warmup_bwd
print(f"  Backward JIT compile + first eval: {t_warmup_bwd:.2f}s", flush=True)

t_bwd_runs = []
for _ in range(N_TIMING):
    t0 = time.time()
    _ = jax.block_until_ready(_loss_and_grad(theta_star))
    t_bwd_runs.append(time.time() - t0)
T_backward_s = float(np.median(t_bwd_runs))
print(f"  T_backward (median of {N_TIMING} runs): {T_backward_s:.4f}s", flush=True)

T_ratio = T_backward_s / T_forward_s
crossover_p = T_ratio / 2.0
print(f"\n  T_ratio = T_backward / T_forward = {T_ratio:.2f}", flush=True)
print(f"  AD beats FD when p > {crossover_p:.1f} parameters", flush=True)


# ── Timing sweep: T_ad vs T_fd(p) ────────────────────────────────────────────
print("\n=== Timing sweep: cost vs p ===", flush=True)
p_values = [1, 2, 5, 10, 20, 50]
# T_ad is constant (one backward pass)
T_ad_s = [T_backward_s] * len(p_values)
# T_fd(p) = 2p * T_forward (central differences)
T_fd_s = [2 * pv * T_forward_s for pv in p_values]
for pv, tad, tfd in zip(p_values, T_ad_s, T_fd_s):
    print(f"  p={pv:3d}: T_AD={tad:.4f}s  T_FD={tfd:.4f}s  speedup={tfd/tad:.1f}x", flush=True)


# ── Initial parameter perturbation ───────────────────────────────────────────
rng = np.random.default_rng(42)
theta_0_np = rng.uniform(0.7, 1.3, size=P)
theta_0 = jnp.array(theta_0_np, dtype=jnp.float64)

loss_0 = float(loss_fn(theta_0))
print(f"\n  Initial theta_0: {theta_0_np.tolist()}", flush=True)
print(f"  Initial loss: {loss_0:.4e}", flush=True)
print(f"  Initial ||theta_0 - theta_star||_2 = {float(jnp.linalg.norm(theta_0 - theta_star)):.4f}", flush=True)


# ── Adam optimizer (manual, no optax required) ────────────────────────────────

def adam_step(params, m, v, grad, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1.0 - b1) * grad
    v = b2 * v + (1.0 - b2) * grad ** 2
    m_hat = m / (1.0 - b1 ** t)
    v_hat = v / (1.0 - b2 ** t)
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, m, v


print("\n" + "=" * 60, flush=True)
print("=== Method A: Adam + jax.grad (200 steps) ===", flush=True)
print("=" * 60, flush=True)

LR_ADAM = 0.02
N_ADAM  = 200

theta   = theta_0
m       = jnp.zeros(P, dtype=jnp.float64)
v       = jnp.zeros(P, dtype=jnp.float64)

adam_loss_history       = []
adam_param_err_history  = []
adam_grad_evals         = []   # cumulative number of gradient evaluations
adam_time_s             = []   # cumulative wall-clock seconds

t_adam_start = time.time()
n_grad_evals = 0

for step in range(1, N_ADAM + 1):
    t_step = time.time()
    loss_val, g = _loss_and_grad(theta)
    jax.block_until_ready((loss_val, g))
    n_grad_evals += 1

    theta, m, v = adam_step(theta, m, v, g, step, lr=LR_ADAM)

    loss_f   = float(loss_val)
    err_f    = float(jnp.linalg.norm(theta - theta_star))
    t_cumul  = time.time() - t_adam_start

    adam_loss_history.append(loss_f)
    adam_param_err_history.append(err_f)
    adam_grad_evals.append(n_grad_evals)
    adam_time_s.append(t_cumul)

    if step % 20 == 0 or step == 1:
        print(
            f"  Adam step {step:3d}: loss={loss_f:.4e}  "
            f"||theta-theta*||={err_f:.4f}  "
            f"({time.time()-t_step:.2f}s/step)",
            flush=True,
        )

t_adam_total = time.time() - t_adam_start
print(f"\n  Adam finished in {t_adam_total:.1f}s  ({n_grad_evals} grad evals)", flush=True)
print(f"  Final loss: {adam_loss_history[-1]:.4e}", flush=True)
print(f"  Final ||theta-theta*||_2: {adam_param_err_history[-1]:.4f}", flush=True)


# ── Nelder-Mead ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method B: Nelder-Mead (gradient-free) ===", flush=True)
print("=" * 60, flush=True)

from scipy.optimize import minimize as _sp_minimize

nm_loss_history = []
nm_nfev_history = []
_nm_evals = [0]
_t_nm_start = time.time()


def _loss_np_nm(x):
    _nm_evals[0] += 1
    l = float(loss_fn(jnp.array(x, dtype=jnp.float64)))
    nm_loss_history.append(l)
    nm_nfev_history.append(_nm_evals[0])
    if _nm_evals[0] % 50 == 0 or _nm_evals[0] == 1:
        print(f"  NM eval {_nm_evals[0]:4d}: loss={l:.4e}  "
              f"({time.time()-_t_nm_start:.1f}s elapsed)", flush=True)
    return l


_nm_result = _sp_minimize(
    _loss_np_nm,
    x0=np.array(theta_0_np),
    method="Nelder-Mead",
    options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True},
)
t_nm_total = time.time() - _t_nm_start
print(f"\n  Nelder-Mead finished: {_nm_result.message}  ({_nm_evals[0]} evals, {t_nm_total:.1f}s)", flush=True)
print(f"  Final loss: {float(_nm_result.fun):.4e}", flush=True)
nm_theta_final = jnp.array(_nm_result.x, dtype=jnp.float64)
print(f"  Final ||theta-theta*||_2: {float(jnp.linalg.norm(nm_theta_final - theta_star)):.4f}", flush=True)


# ── L-BFGS-B with FD gradient ─────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method C: L-BFGS-B + FD gradient (scipy jac=None) ===", flush=True)
print("=" * 60, flush=True)

lbfgsb_loss_history = []
lbfgsb_nfev_history = []
_lbfgsb_evals = [0]
_t_lbfgsb_start = time.time()


def _loss_np_lbfgsb(x):
    _lbfgsb_evals[0] += 1
    l = float(loss_fn(jnp.array(x, dtype=jnp.float64)))
    lbfgsb_loss_history.append(l)
    lbfgsb_nfev_history.append(_lbfgsb_evals[0])
    if _lbfgsb_evals[0] % 20 == 0 or _lbfgsb_evals[0] == 1:
        print(f"  L-BFGS-B eval {_lbfgsb_evals[0]:4d}: loss={l:.4e}  "
              f"({time.time()-_t_lbfgsb_start:.1f}s elapsed)", flush=True)
    return l


_lbfgsb_result = _sp_minimize(
    _loss_np_lbfgsb,
    x0=np.array(theta_0_np),
    method="L-BFGS-B",
    jac=None,          # scipy uses FD; each gradient step = 2p+1 forward evals
    options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-8},
)
t_lbfgsb_total = time.time() - _t_lbfgsb_start
print(f"\n  L-BFGS-B finished: {_lbfgsb_result.message}  ({_lbfgsb_evals[0]} evals, {t_lbfgsb_total:.1f}s)", flush=True)
print(f"  Final loss: {float(_lbfgsb_result.fun):.4e}", flush=True)
lbfgsb_theta_final = jnp.array(_lbfgsb_result.x, dtype=jnp.float64)
print(f"  Final ||theta-theta*||_2: {float(jnp.linalg.norm(lbfgsb_theta_final - theta_star)):.4f}", flush=True)


# ── Save results ──────────────────────────────────────────────────────────────
print("\n=== Saving results ===", flush=True)

# Convert forward-equivalent counts for Adam:
# Each grad eval costs T_backward = T_ratio * T_forward forward-equivalents
adam_fwd_equiv = [int(g) * T_ratio for g in adam_grad_evals]

results = {
    "p": P,
    "theta_star": theta_star.tolist(),
    "theta_0": theta_0_np.tolist(),
    "GPP_target": float(GPP_target),
    "T_forward_s": T_forward_s,
    "T_backward_s": T_backward_s,
    "T_ratio": T_ratio,
    "crossover_p": crossover_p,
    "adam": {
        "loss_history":        adam_loss_history,
        "param_err_history":   adam_param_err_history,
        "grad_evals":          adam_grad_evals,
        "fwd_equiv":           adam_fwd_equiv,
        "time_s":              adam_time_s,
        "final_loss":          adam_loss_history[-1],
        "final_param_err":     adam_param_err_history[-1],
        "total_time_s":        t_adam_total,
    },
    "nelder_mead": {
        "loss_history":    nm_loss_history,
        "nfev_history":    nm_nfev_history,
        "final_loss":      float(_nm_result.fun),
        "final_param_err": float(jnp.linalg.norm(nm_theta_final - theta_star)),
        "time_s":          t_nm_total,
        "converged":       bool(_nm_result.success),
    },
    "lbfgsb_fd": {
        "loss_history":    lbfgsb_loss_history,
        "nfev_history":    lbfgsb_nfev_history,
        "final_loss":      float(_lbfgsb_result.fun),
        "final_param_err": float(jnp.linalg.norm(lbfgsb_theta_final - theta_star)),
        "time_s":          t_lbfgsb_total,
        "converged":       bool(_lbfgsb_result.success),
    },
    "timing_sweep": {
        "p_values": p_values,
        "T_ad_s":   T_ad_s,
        "T_fd_s":   T_fd_s,
    },
}

out_path = OUTPUT_DIR / "multipar_calibration_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}", flush=True)

print("\n=== multipar_calibration.py complete ===", flush=True)
