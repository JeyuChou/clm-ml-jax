#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=param-sens
#SBATCH --output=logs/%j_param_sensitivity.out
#SBATCH --error=logs/%j_param_sensitivity.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=hpc_test
#SBATCH --constraint=a100

# Parameter sensitivity analysis for CLM-ML-JAX.
# Computes dGPP/dθ and dLE/dθ for alpha_sw, alpha_tref, alpha_vcmax25, alpha_iota
# at the CHATS7 operating point.
#
# Output: logs/${JOBID}_param_sensitivity.{out,err}
#         diags/figures/param_sensitivity.png

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

# ── Run parameter sensitivity analysis ───────────────────────────────────────
echo ""
echo "=== Parameter sensitivity analysis (CHATS7, Euler 1 sub-step) ==="
cd /burg-archive/home/al4385/clm-ml-jax/src
CLM_ML_NO_CHECKPOINT=1 python ../diags/param_sensitivity.py

echo ""
echo "=== run_param_sensitivity.sh complete ==="
