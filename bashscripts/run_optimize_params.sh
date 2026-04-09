#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=opt-params
#SBATCH --output=logs/%j_optimize_params.out
#SBATCH --error=logs/%j_optimize_params.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --constraint=a100

# Gradient-based parameter optimization for CLM-ML-JAX.
# Phase 1: vcmaxpft recovery from synthetic CHATS7 observations.
# See diags/optimize_params.py for full documentation.
#
# Output: logs/${JOBID}_optimize_params.{out,err}
#         diags/figures/optimization_vcmax25.png
#         diags/figures/optimization_vcmax25_results.json

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

# ── Move to project root ──────────────────────────────────────────────────────
cd /burg-archive/home/al4385/clm-ml-jax

# ── Run optimization: synthetic case (vcmaxpft recovery) ──────────────────────
echo ""
echo "=== Phase 1: vcmaxpft identifiability test (synthetic CHATS7 obs) ==="
cd /burg-archive/home/al4385/clm-ml-jax/src
CLM_ML_NO_CHECKPOINT=1 python ../diags/optimize_params.py \
    --synthetic --vcmax-true 125.0 --n-steps 100

echo ""
echo "=== run_optimize_params.sh complete ==="
