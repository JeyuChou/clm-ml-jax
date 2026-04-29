#!/bin/bash
#SBATCH --job-name=multipar-calib
#SBATCH --output=/burg-archive/home/al4385/clm-ml-jax/logs/%j_multipar_calibration.out
#SBATCH --error=/burg-archive/home/al4385/clm-ml-jax/logs/%j_multipar_calibration.err
#SBATCH --partition=glab1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=glab

# Multi-parameter calibration experiment (p=10, all active): AD vs FD vs Nelder-Mead.
#
# Redesigned experiment (Session 42):
#   - Multi-timestep loss: T=8 midday steps spread across May — breaks equifinality
#     (single-step was under-determined: 10 params, 3 outputs → equifinal solutions)
#   - Adam: cosine-annealing LR 0.01→1e-4 over 500 steps (was fixed 0.005, oscillated)
#   - Combined loss: normalized MSE over (GPP, H, LE) — all params have active gradients
#   - p=10 all active: SW split into 4 waveband components
#   - 4 methods: Adam/AD, L-BFGS-B/AD (scipy jac=True), L-BFGS-B/FD, Nelder-Mead
#
# Outputs:
#   diags/output/multipar_calibration_results.json
#   diags/figures/multipar_calibration.{pdf,png}

nvidia-smi

module load anaconda
module load shared
module load cuda12.8/toolkit/12.8.61

source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax

SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

# ── JAX compilation cache ───────────────────────────────────────────────────
export JAX_COMPILATION_CACHE_DIR=/burg-archive/home/al4385/.cache/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache dir: $JAX_COMPILATION_CACHE_DIR"

cd /burg-archive/home/al4385/clm-ml-jax

echo "=== multipar_calibration.py (p=10, AD vs FD vs Nelder-Mead) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/multipar_calibration.py

echo "=== plot_multipar_calibration.py (publication figure) ==="
python diags/plot_multipar_calibration.py

echo "=== run_multipar_calibration.sh complete ==="
