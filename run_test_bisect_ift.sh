#!/bin/bash
#SBATCH --job-name=bisect-ift
#SBATCH --output=logs/%j_bisect_ift.out
#SBATCH --error=logs/%j_bisect_ift.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=glab

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

cd /burg-archive/home/al4385/clm-ml-jax
CLM_ML_NO_CHECKPOINT=1 python diags/test_bisect_ift.py
