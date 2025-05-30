#!/bin/bash

#SBATCH --job-name=abm-analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
else
    echo "WARNING: Could not find conda.sh, using module python directly"
    # If conda environment isn't available, just use the module python
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create analysis results directory
mkdir -p analysis_results
mkdir -p logs

# Print job information
echo "============================================"
echo "ABM Analysis Job Information"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
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

# Run quick analysis (always works)
echo "Starting quick analysis..."
python quick_analysis.py \
    --csv_file "$CSV_FILE" \
    --output_dir analysis_results

QUICK_EXIT_CODE=$?

if [ $QUICK_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Quick analysis completed!"
else
    echo "ERROR: Quick analysis failed with exit code $QUICK_EXIT_CODE"
fi

echo "============================================"

# Check if Spark is available and try Spark analysis
if command -v spark-submit &> /dev/null; then
    echo "Spark found. Running Spark analysis..."
    
    # Load Spark module if available
    module load spark/3.3.2 2>/dev/null || echo "Spark module not available, using default"
    
    # Set Spark environment
    export PYSPARK_DRIVER_PYTHON=$(which python)
    export PYSPARK_PYTHON=$(which python)
    
    # Run Spark analysis with reduced resources
    spark-submit \
        --total-executor-cores 3 \
        --executor-memory 2G \
        --driver-memory 2G \
        --conf spark.sql.adaptive.enabled=true \
        --conf spark.sql.adaptive.coalescePartitions.enabled=true \
        spark_analysis.py \
        --csv_file "$CSV_FILE" \
        --output_dir analysis_results
    
    SPARK_EXIT_CODE=$?
    
    if [ $SPARK_EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Spark analysis completed!"
    else
        echo "WARNING: Spark analysis failed with exit code $SPARK_EXIT_CODE"
        echo "Quick analysis results are still available."
    fi
else
    echo "Spark not available. Only quick analysis was performed."
fi

echo "============================================"
echo "Analysis completed at: $(date)"

# Show results
echo "Analysis results:"
ls -la analysis_results/

# Show generated plots
echo "Generated plots:"
find analysis_results/ -name "*.png" -exec basename {} \;

echo "============================================"
echo "Analysis job completed!"