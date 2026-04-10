#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=multisite-vmap
#SBATCH --output=logs/%j_multisite_benchmark.out
#SBATCH --error=logs/%j_multisite_benchmark.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=hpc_test
#SBATCH --constraint=a100

# ── Load CUDA toolkit ─────────────────────────────────────────────────────────
module load cuda12.8/toolkit/12.8.61

# ── Activate conda environment ─────────────────────────────────────────────────
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax

# ── Expose JAX's bundled CUDA libraries ───────────────────────────────────────
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

# ── JAX compilation cache (eliminates 290s recompile on subsequent runs) ──────
export JAX_COMPILATION_CACHE_DIR=/burg-archive/home/al4385/.cache/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache dir: $JAX_COMPILATION_CACHE_DIR"

# ── Move to project root ──────────────────────────────────────────────────────
cd /burg-archive/home/al4385/clm-ml-jax

# ── Run benchmark: full RK4 physics, GPU + CPU, N = 1,2,4,8,16,32 ────────────
echo ""
echo "=== Multi-site vmap benchmark — FULL RK4 (GPU) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/benchmark_multisite.py \
    --n-sites 1,2,4,8,16,32 \
    --repeats 3 \
    --backend gpu \
    --full-physics

echo ""
echo "=== Multi-site vmap benchmark — FULL RK4 (CPU) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/benchmark_multisite.py \
    --n-sites 1,2,4,8 \
    --repeats 2 \
    --backend cpu \
    --full-physics

echo ""
echo "=== run_multisite_benchmark.sh complete ==="
