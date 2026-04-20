#!/bin/bash
#SBATCH --job-name=ensemble-bench
#SBATCH --output=/burg-archive/home/al4385/clm-ml-jax/logs/%j_ensemble_benchmark.out
#SBATCH --error=/burg-archive/home/al4385/clm-ml-jax/logs/%j_ensemble_benchmark.err
#SBATCH --partition=glab1
#SBATCH --time=1-20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=glab

# Parameter ensemble GPU vs CPU benchmark.
#
# Runs CLM-ML-JAX forward model for N parameter samples (N=1..2048) via
# jax.vmap on GPU and CPU, comparing vmap throughput vs sequential baseline.
#
# Key experiment: GPU is slower than CPU for N=1 (single column, small arrays)
# but increasingly faster for large N as vmap exploits GPU parallelism.
# Expected crossover ~N=32-64; expected GPU advantage at N=1024: ~60x over
# CPU sequential.
#
# 5 scale parameters uniform [0.8,1.2]: Vcmax25, T_air, SW_rad, q_ref, dpai
# Outputs per sample: [GPP, H, LE]
#
# Estimated runtime:
#   - JIT compile (first call): ~700-1200s
#   - GPU vmap N=1..2048: ~2-5min total
#   - CPU vmap N=1..2048: ~10-20min total
#   - CPU sequential N<=64: ~5min total
#   Total: ~1-2h
#
# Outputs:
#   diags/figures/ensemble_benchmark.csv
#   diags/figures/ensemble_benchmark.png

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

export CLM_ML_NO_CHECKPOINT=1

cd /burg-archive/home/al4385/clm-ml-jax

echo "=== benchmark_ensemble.py (GPU vs CPU vmap ensemble N=1..2048) ==="
python diags/benchmark_ensemble.py

echo "=== run_ensemble_benchmark.sh complete ==="
