#!/bin/bash

#SBATCH --job-name=abm-scalability-benchmark
#SBATCH --output=logs/scalability_%j.out
#SBATCH --error=logs/scalability_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=caslake
#SBATCH --account=macs30123

# Load required modules
module load python/anaconda-2022.05

# Initialize conda properly
CONDA_SH=$(find /software -name "conda.sh" 2>/dev/null | head -n 1)
if [ -z "$CONDA_SH" ]; then
    CONDA_SH="/software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh"
fi

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate abm_env
    echo "Using conda environment: abm_env"
else
    echo "Using module python directly"
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create benchmarks directory
mkdir -p benchmarks
mkdir -p logs

# Print job information
echo "============================================"
echo "ABM Scalability Benchmark"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Available cores: $SLURM_NTASKS"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "============================================"

# Install additional packages if needed
echo "Installing additional packages for benchmarking..."
pip install --user pyarrow fastparquet tables || echo "Some packages may not install, continuing..."

echo "============================================"

# Run comprehensive scalability benchmark
echo "Starting comprehensive scalability benchmark..."
echo "Testing compute scalability (cores: 1, 2, 4, 8, 16, 20)"
echo "Testing storage formats: CSV, Pickle, Parquet, HDF5, Feather"
echo "Data sizes: 1K, 10K, 100K, 1M rows"

python benchmark_scalability.py \
    --max_sims 400 \
    --max_cores 20 \
    --output_dir benchmarks

BENCHMARK_EXIT_CODE=$?

echo "============================================"
echo "Benchmark completed at: $(date)"
echo "Exit code: $BENCHMARK_EXIT_CODE"

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Scalability benchmark completed!"
    
    # Show results
    echo "Benchmark results:"
    ls -la benchmarks/
    
    # Show key findings
    echo "Generated files:"
    find benchmarks/ -name "*.png" -exec echo "  Plot: {}" \;
    find benchmarks/ -name "*.json" -exec echo "  Data: {}" \;
    find benchmarks/ -name "*summary*.csv" -exec echo "  Summary: {}" \;
    
    # Show size of benchmark data
    echo "Total benchmark data size:"
    du -sh benchmarks/
    
    # Show quick summary
    echo "Performance Summary:"
    if [ -f benchmarks/scalability_summary_*.csv ]; then
        echo "Latest summary:"
        tail -n 10 benchmarks/scalability_summary_*.csv
    fi
else
    echo "ERROR: Scalability benchmark failed with exit code $BENCHMARK_EXIT_CODE"
fi

echo "============================================"
echo "Scalability benchmark job completed!"