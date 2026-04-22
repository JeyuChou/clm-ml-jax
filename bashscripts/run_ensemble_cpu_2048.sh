#!/bin/bash
#SBATCH --job-name=cpu-vmap-2048
#SBATCH --output=/burg-archive/home/al4385/clm-ml-jax/logs/%j_ensemble_cpu_2048.out
#SBATCH --error=/burg-archive/home/al4385/clm-ml-jax/logs/%j_ensemble_cpu_2048.err
#SBATCH --partition=glab1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=glab

# CPU-only vmap benchmark for N=2048.
# No GPU requested — pure CPU compilation benchmark.
# Expected compile time: hours (N=512 ~6-10h, N=1024 ~12-24h, N=2048 ~24-72h).

module load anaconda
module load shared
module load cuda12.8/toolkit/12.8.61

source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax

SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':'):$LD_LIBRARY_PATH
fi

python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

export JAX_COMPILATION_CACHE_DIR=/burg-archive/home/al4385/.cache/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache dir: $JAX_COMPILATION_CACHE_DIR"

export CLM_ML_NO_CHECKPOINT=1

cd /burg-archive/home/al4385/clm-ml-jax

echo "=== benchmark_ensemble_cpu_2048.py ==="
python diags/benchmark_ensemble_cpu_2048.py

echo "=== run_ensemble_cpu_2048.sh complete ==="
