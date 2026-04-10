#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=aot-compile
#SBATCH --output=logs/%j_aot_compile.out
#SBATCH --error=logs/%j_aot_compile.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --constraint=a100

# ── Load modules ──────────────────────────────────────────────────────────────
module load cuda12.8/toolkit/12.8.61
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax

# ── Expose JAX's bundled CUDA libraries ───────────────────────────────────────
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

# ── JAX compilation cache ─────────────────────────────────────────────────────
export JAX_COMPILATION_CACHE_DIR=/burg-archive/home/al4385/.cache/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache dir: $JAX_COMPILATION_CACHE_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

# ── Run AOT compilation ───────────────────────────────────────────────────────
cd /burg-archive/home/al4385/clm-ml-jax

echo ""
echo "=== AOT compile: warm up persistent JIT cache ==="
CLM_ML_NO_CHECKPOINT=1 python diags/aot_compile.py --export

echo ""
echo "=== run_aot_compile.sh complete ==="
