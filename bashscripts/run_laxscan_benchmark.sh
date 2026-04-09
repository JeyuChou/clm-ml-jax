#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=laxscan-bench
#SBATCH --output=logs/%j_laxscan_benchmark.out
#SBATCH --error=logs/%j_laxscan_benchmark.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=hpc_test
#SBATCH --constraint=a100

module load cuda12.8/toolkit/12.8.61
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax

SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

cd /burg-archive/home/al4385/clm-ml-jax

echo ""
echo "=== lax.scan benchmark: correctness + runtime ==="
CLM_ML_NO_CHECKPOINT=1 python diags/benchmark_laxscan.py

echo ""
echo "=== run_laxscan_benchmark.sh complete ==="
