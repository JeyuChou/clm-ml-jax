#!/bin/bash
#SBATCH --job-name=photo-kernel
#SBATCH --output=logs/%j_photo_kernel_grad.out
#SBATCH --error=logs/%j_photo_kernel_grad.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000
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
CLM_ML_NO_CHECKPOINT=1 python diags/test_photo_kernel_grad.py
