#!/bin/bash

#SBATCH --job-name=abm-spark-analysis
#SBATCH --output=logs/spark_analysis_%j.out
#SBATCH --error=logs/spark_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --partition=caslake
#SBATCH --account=macs30123

# Load required modules
module load python/anaconda-2022.05 spark/3.3.2

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

# Set Python and Spark environment
export PYSPARK_DRIVER_PYTHON=$(which python)
export PYSPARK_PYTHON=$(which python)

# Create analysis results directory
mkdir -p analysis_results
mkdir -p logs

# Print job information
echo "============================================"
echo "ABM Spark Analysis Job Information"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "============================================"

# Find the most recent CSV results file
CSV_FILE=$(ls -t results/abm_sweep_results_*.csv 2>/dev/null | head -n 1)

if [ -z "$CSV_FILE" ]; then
    echo "ERROR: No CSV results file found in results/ directory"
    echo "Available files in results/:"
    ls -la results/ || echo "Results directory does not exist"
    exit 1
fi

echo "Using results file: $CSV_FILE"
echo "File size: $(du -h "$CSV_FILE" | cut -f1)"
echo "Number of rows: $(wc -l < "$CSV_FILE")"
echo "============================================"

# Run Spark analysis
echo "Starting Spark analysis..."
spark-submit \
    --total-executor-cores 9 \
    --executor-memory 4G \
    --driver-memory 4G \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    spark_analysis.py \
    --csv_file "$CSV_FILE" \
    --output_dir analysis_results

SPARK_EXIT_CODE=$?

echo "============================================"
echo "Spark analysis completed at: $(date)"
echo "Exit code: $SPARK_EXIT_CODE"

if [ $SPARK_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Spark analysis completed successfully!"
    
    # Show results
    echo "Analysis results:"
    ls -la analysis_results/
    
    # Show generated files
    echo "Generated files:"
    find analysis_results/ -name "*.png" -exec echo "  Plot: {}" \;
    find analysis_results/ -name "*.json" -exec echo "  Data: {}" \;
    find analysis_results/ -name "*.md" -exec echo "  Report: {}" \;
    
    # Show disk usage
    echo "Total analysis results size:"
    du -sh analysis_results/
else
    echo "ERROR: Spark analysis failed with exit code $SPARK_EXIT_CODE"
    echo "Check the error log for details."
    
    # Try to run quick analysis as fallback
    echo "Attempting quick analysis as fallback..."
    python quick_analysis.py \
        --csv_file "$CSV_FILE" \
        --output_dir analysis_results
    
    QUICK_EXIT_CODE=$?
    if [ $QUICK_EXIT_CODE -eq 0 ]; then
        echo "Quick analysis completed successfully as fallback."
    else
        echo "Both Spark and quick analysis failed."
    fi
fi

echo "============================================"
echo "Analysis job completed!"