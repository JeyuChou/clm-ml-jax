"""
Direct photosynthesis kernel gradient test.

Tests d(agross)/d(apar) for a single canopy layer by calling
the vmapped photosynthesis kernel directly — bypasses MLCanopyFluxes
and SolarRadiation, so any gradient error here is PURELY in the
photosynthesis kernel.

If this test PASSES: the bug is NOT in the photosynthesis kernel.
If this test FAILS: the bug IS in the photosynthesis kernel.

Usage:
    cd /burg-archive/home/al4385/clm-ml-jax
    CLM_ML_NO_CHECKPOINT=1 python diags/test_photo_kernel_grad.py
"""
from __future__ import annotations
import os, sys
os.environ['CLM_ML_NO_CHECKPOINT'] = '1'

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diags.expt_init import (
    mlcanopy_inst, grid, jax, jnp,
)
from multilayer_canopy.MLclm_varpar import isun, isha
from multilayer_canopy.MLclm_varctl import gs_type, colim_type
from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _get_vmapped_photo_kernel, _make_leaf_photo_kernel,
)
from multilayer_canopy import MLclm_varpar as _vpar
from multilayer_canopy import pftconMod
import numpy as np

_p    = grid.p
_ncan = grid.ncan

EPS = 1e-4

print("\n" + "="*70)
print("PHOTOSYNTHESIS KERNEL GRADIENT TEST")
print(f"  gs_type={gs_type}, colim_type={colim_type}, ncan={_ncan}")
print("="*70)

# ── Extract baseline leaf-level data from warmed-up state ─────────────────────
_sl = slice(1, _ncan + 1)

# Extract data for sunlit leaves (il=isun=0)
for il, leaf_label in [(isun, 'isun'), (isha, 'isha')]:
    print(f"\n--- Leaf type: {leaf_label} ---")

    dpai_arr   = np.asarray(mlcanopy_inst.dpai_profile[_p, _sl])        # (ncan,)
    tleaf_arr  = np.asarray(mlcanopy_inst.tleaf_leaf[_p, _sl, il])      # (ncan,)
    vcmax25_arr= np.asarray(mlcanopy_inst.vcmax25_leaf[_p, _sl, il])
    jmax25_arr = np.asarray(mlcanopy_inst.jmax25_leaf[_p, _sl, il])
    rd25_arr   = np.asarray(mlcanopy_inst.rd25_leaf[_p, _sl, il])
    kp25_arr   = np.asarray(mlcanopy_inst.kp25_leaf[_p, _sl, il])
    eair_arr   = np.asarray(mlcanopy_inst.eair_profile[_p, _sl])
    apar_arr   = np.asarray(mlcanopy_inst.apar_leaf[_p, _sl, il])       # (ncan,)
    gbc_arr    = np.asarray(mlcanopy_inst.gbc_leaf[_p, _sl, il])
    gbv_arr    = np.asarray(mlcanopy_inst.gbv_leaf[_p, _sl, il])
    cair_arr   = np.asarray(mlcanopy_inst.cair_profile[_p, _sl])
    o2ref_py   = float(mlcanopy_inst.o2ref_forcing[_p])

    # PFT for patch p
    pft_idx = int(mlcanopy_inst.pft_canopy[_p])

    # Stomatal parameters
    if gs_type == 0:
        g0_val = float(pftconMod._g0_MED_np[pft_idx])
        g1_val = float(pftconMod._g1_MED_np[pft_idx])
    elif gs_type == 1:
        g0_val = float(pftconMod._g0_BB_np[pft_idx])
        g1_val = float(pftconMod._g1_BB_np[pft_idx])
    else:
        print("WUE gs_type not supported in this test"); continue

    # Photosynthesis parameters from pftconMod
    c3psn_val  = float(pftconMod.c3psn[pft_idx])
    is_c3_bool = round(c3psn_val) == 1
    vcmaxha    = float(pftconMod.vcmaxha[pft_idx])
    vcmaxhd    = float(pftconMod.vcmaxhd[pft_idx])
    vcmaxse    = float(pftconMod.vcmaxse[pft_idx])
    vcmaxc     = float(pftconMod.vcmaxc[pft_idx])
    jmaxha     = float(pftconMod.jmaxha[pft_idx])
    jmaxhd     = float(pftconMod.jmaxhd[pft_idx])
    jmaxse     = float(pftconMod.jmaxse[pft_idx])
    jmaxc      = float(pftconMod.jmaxc[pft_idx])
    rdc        = float(pftconMod.rdc[pft_idx])

    vmapped = _get_vmapped_photo_kernel(
        is_c3=is_c3_bool,
        c3psn_pft_val=c3psn_val,
        vcmaxha=vcmaxha, vcmaxhd=vcmaxhd,
        vcmaxse=vcmaxse, vcmaxc=vcmaxc,
        jmaxha=jmaxha, jmaxhd=jmaxhd,
        jmaxse=jmaxse, jmaxc=jmaxc,
        rdc=rdc,
        g0_val=g0_val, g1_val=g1_val,
        o2ref_p=o2ref_py,
    )

    # Convert to JAX arrays
    dpai_j   = jnp.array(dpai_arr)
    tleaf_j  = jnp.array(tleaf_arr)
    vcmax25_j= jnp.array(vcmax25_arr)
    jmax25_j = jnp.array(jmax25_arr)
    rd25_j   = jnp.array(rd25_arr)
    kp25_j   = jnp.array(kp25_arr)
    eair_j   = jnp.array(eair_arr)
    apar_j   = jnp.array(apar_arr)
    gbc_j    = jnp.array(gbc_arr)
    gbv_j    = jnp.array(gbv_arr)
    cair_j   = jnp.array(cair_arr)

    def forward_agross_kernel(apar_scale: jnp.ndarray) -> jnp.ndarray:
        """Scale all apar values by apar_scale, return sum(agross)."""
        layer_out = vmapped(
            dpai_j, tleaf_j, vcmax25_j, jmax25_j, rd25_j, kp25_j,
            eair_j, apar_j * apar_scale, gbc_j, gbv_j, cair_j,
        )
        # layer_out[0] = agross (shape: ncan)
        return jnp.sum(layer_out[0])

    # Baseline
    baseline_agross = float(forward_agross_kernel(jnp.float64(1.0)))
    print(f"  baseline sum(agross) = {baseline_agross:.6f}", flush=True)

    # JAX gradient
    import time
    t0 = time.time()
    jax_grad = float(jax.jit(jax.grad(forward_agross_kernel))(jnp.float64(1.0)))
    print(f"  JAX grad d(sum(agross))/d(scale) = {jax_grad:.6e}  ({time.time()-t0:.1f}s)", flush=True)

    # FD
    f_plus  = float(forward_agross_kernel(jnp.float64(1.0 + EPS)))
    f_minus = float(forward_agross_kernel(jnp.float64(1.0 - EPS)))
    fd_grad = (f_plus - f_minus) / (2.0 * EPS)
    print(f"  FD  grad d(sum(agross))/d(scale) = {fd_grad:.6e}", flush=True)

    rel = abs(jax_grad - fd_grad) / (abs(fd_grad) + 1e-30)
    print(f"  Rel error: {rel:.3e}  {'PASS' if rel < 0.01 else 'FAIL'}", flush=True)

    # Also show baseline sum(apar) for comparison
    print(f"  baseline sum(apar) = {float(jnp.sum(apar_j)):.6f}", flush=True)
    print(f"  FD ≈ d(agross*apar)/d(apar) expected ≠ sum(apar) [nonlinear]", flush=True)

    # Layer-by-layer breakdown for first few layers
    print(f"\n  Layer-by-layer (first 5):", flush=True)
    print(f"  {'ic':>3}  {'apar':>10}  {'agross_jax':>12}  {'agross_base':>12}", flush=True)
    for ic in range(min(5, _ncan)):
        _apar_ic = float(apar_j[ic])
        _agross_base = float(forward_agross_kernel(jnp.float64(1.0))) # could extract differently
        # JAX grad for single layer contribution
        def _single_layer(s):
            _a = apar_j.at[ic].set(apar_j[ic] * s)
            out = vmapped(dpai_j, tleaf_j, vcmax25_j, jmax25_j, rd25_j, kp25_j, eair_j, _a, gbc_j, gbv_j, cair_j)
            return out[0][ic]
        _g_ic = float(jax.grad(_single_layer)(jnp.float64(1.0)))
        _fd_ic = (float(_single_layer(jnp.float64(1.0+EPS))) - float(_single_layer(jnp.float64(1.0-EPS)))) / (2*EPS)
        print(f"  {ic+1:>3}  {_apar_ic:>10.4f}  {_g_ic:>12.6e}  {_fd_ic:>12.6e}", flush=True)

print("\n=== test_photo_kernel_grad.py complete ===", flush=True)
