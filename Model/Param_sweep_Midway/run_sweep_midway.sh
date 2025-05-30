#!/bin/bash

#SBATCH --job-name=abm-param-sweep
#SBATCH --output=logs/abm_sweep_%j.out
#SBATCH --error=logs/abm_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=caslake
#SBATCH --account=macs30123

# Load required modules
module load python/anaconda-2022.05

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate abm_env

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create results directory if it doesn't exist
mkdir -p results
mkdir -p logs

# Print job information
echo "============================================"
echo "ABM Parameter Sweep Job Information"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "============================================"

# Print system information
echo "Available CPU cores: $(nproc)"
echo "Memory info:"
free -h
echo "============================================"

# Run the parameter sweep
echo "Starting ABM parameter sweep..."
echo "Parameters:"
echo "  - Informality rates: 0.1 to 0.8 (8 steps)"
echo "  - Seeds per rate: 100"
echo "  - Total simulations: 800"
echo "  - Processes: 20"
echo "============================================"

python param_sweep.py \
    --min_informality 0.1 \
    --max_informality 0.8 \
    --informality_steps 8 \
    --n_seeds 100 \
    --n_processes 20 \
    --output_dir results

EXIT_CODE=$?

echo "============================================"
echo "Parameter sweep completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Parameter sweep completed successfully!"
    
    # Check results
    echo "Results directory contents:"
    ls -la results/
    
    # Show disk usage
    echo "Disk usage of results:"
    du -sh results/
    
    # Count files
    echo "Number of result files:"
    ls results/ | wc -l
    
    # Show recent CSV file info
    LATEST_CSV=$(ls -t results/abm_sweep_results_*.csv 2>/dev/null | head -n 1)
    if [ ! -z "$LATEST_CSV" ]; then
        echo "Latest CSV file: $LATEST_CSV"
        echo "CSV file size: $(du -h "$LATEST_CSV" | cut -f1)"
        echo "CSV rows: $(wc -l < "$LATEST_CSV")"
    fi
else
    echo "ERROR: Parameter sweep failed with exit code $EXIT_CODE"
    echo "Check the error log for details."
fi

echo "============================================"
echo "Job completed!"
