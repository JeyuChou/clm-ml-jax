#!/bin/bash
#SBATCH --job-name=le-h-grad-check
#SBATCH --output=logs/%j_le_h_grad_check.out
#SBATCH --error=logs/%j_le_h_grad_check.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=glab

# Runs LE and H gradient checks for all 5 parameters.
# Each jax.grad compile takes ~800-1200s for new kernels.
# 5 LE grads + 5 H grads + FD evals = ~10 compiles worst case.
# JAX cache should hit for kernels already compiled by fd_grad_check.
# Time limit: 6h to be safe.

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

echo "=== le_h_grad_check.py (LE + H gradients for all 5 params) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/le_h_grad_check.py

echo "=== sensitivity_analysis.py (Jacobian with fixed Vcmax25 via vcmaxpft_jax) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/sensitivity_analysis.py

echo "=== run_le_h_grad_check.sh complete ==="
