#!/bin/bash
#SBATCH --job-name=dpai-jacobian
#SBATCH --output=logs/%j_dpai_jacobian_check.out
#SBATCH --error=logs/%j_dpai_jacobian_check.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=glab

# Verify dpai gradient fix: sensitivity_analysis.py now scales
# canopystate_inst.elai_patch/esai_patch instead of mlcanopy_inst.dpai_profile
# directly, so the gradient flows through MLCanopyFluxes' dpai recomputation.
# Expected: dpai column no longer zero in Jacobian.

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

export JAX_COMPILATION_CACHE_DIR=/burg-archive/home/al4385/.cache/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache dir: $JAX_COMPILATION_CACHE_DIR"

cd /burg-archive/home/al4385/clm-ml-jax

echo "=== sensitivity_analysis.py (dpai fix: canopystate_inst elai/esai scaling) ==="
CLM_ML_NO_CHECKPOINT=1 python diags/sensitivity_analysis.py

echo "=== dpai_jacobian_check complete ==="
