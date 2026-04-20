#!/bin/bash
#SBATCH --job-name=calib-vcmax-iota
#SBATCH --output=/burg-archive/home/al4385/clm-ml-jax/logs/%j_calibration_vcmax_iota.out
#SBATCH --error=/burg-archive/home/al4385/clm-ml-jax/logs/%j_calibration_vcmax_iota.err
#SBATCH --partition=glab1
#SBATCH --time=1-20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=glab

# Experiment 4 (2-parameter): Joint calibration of Vcmax25 and iota_SPA.
#
# Recovers vcmaxpft[7]=125.0 and iota_SPA[7]=375.0 from CLM defaults 57.7/750.
# Adam (150 steps, log-space) vs Nelder-Mead (300-eval budget).
#
# Each Adam step requires one jax.grad compile the first time (~700-1200s),
# then ~10-30s/step after JIT warmup.  150 steps + NM ≈ 3-5h total.
# Time limit: 1-20h (44h).
#
# Output:
#   diags/figures/calibration_vcmax_iota_convergence.png
#   diags/figures/calibration_vcmax_iota_results.csv

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

echo "=== calibration_vcmax_iota.py (Adam + Nelder-Mead, joint Vcmax25+iota recovery) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/calibration_vcmax_iota.py

echo "=== run_calibration_vcmax_iota.sh complete ==="
