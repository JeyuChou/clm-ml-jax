#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=plot-benchmarks
#SBATCH --output=logs/%j_plot_benchmarks.out
#SBATCH --error=logs/%j_plot_benchmarks.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=short

# CPU-only — just reads CSVs and generates the figure.
# Submit AFTER laxscan_benchmark and multisite_benchmark jobs complete.
# Usage:
#   sbatch --dependency=afterok:<LAXSCAN_JOB>:<MULTISITE_JOB> bashscripts/run_plot_benchmarks.sh

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax

cd /burg-archive/home/al4385/clm-ml-jax

echo "=== Generating benchmark_summary.png ==="
python diags/plot_benchmarks.py

echo "=== run_plot_benchmarks.sh complete ==="
