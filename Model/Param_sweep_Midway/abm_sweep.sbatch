#!/bin/bash

#SBATCH --job-name=abm-parameter-sweep
#SBATCH --output=abm_sweep_%j.out
#SBATCH --error=abm_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=caslake
#SBATCH --account=macs30123

# Load required modules
module load python/anaconda-2022.05

# Initialize conda
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate abm_env

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create results directory
mkdir -p results

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"

# Print system information
echo "Available CPU cores: $(nproc)"
echo "Memory info:"
free -h

# Run the parameter sweep
echo "Starting parameter sweep..."
python param_sweep.py \
    --min_informality 0.1 \
    --max_informality 0.8 \
    --informality_steps 8 \
    --n_seeds 100 \
    --n_processes 20 \
    --output_dir results

echo "Parameter sweep completed at: $(date)"

# Check results
echo "Results directory contents:"
ls -la results/

# Show disk usage
echo "Disk usage of results:"
du -sh results/

echo "Job completed successfully!"
