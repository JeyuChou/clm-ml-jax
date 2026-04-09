#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=calibrat
#SBATCH --output=logs/%j_calibration_demo.out
#SBATCH --error=logs/%j_calibration_demo.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=hpc_test

module load cuda12.8/toolkit/12.8.61

source /burg-archive/home/al4385/clm-ml-jax/.venv/bin/activate

SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

cd /burg-archive/home/al4385/clm-ml-jax
nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

echo "=== Starting calibration_demo.py ==="
cd src
python ../diags/calibration_demo.py
echo "=== calibration_demo.py complete ==="
