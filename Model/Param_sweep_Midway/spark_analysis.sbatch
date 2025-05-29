#!/bin/bash

#SBATCH --job-name=abm-spark-analysis
#SBATCH --output=spark_analysis_%j.out
#SBATCH --error=spark_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --partition=caslake
#SBATCH --account=macs30123

# Load required modules
module load python/anaconda-2022.05 spark/3.3.2

# Set Python and Spark environment
export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

# Create analysis results directory
mkdir -p analysis_results

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Find the most recent CSV results file
CSV_FILE=$(ls -t results/abm_sweep_results_*.csv | head -n 1)

if [ -z "$CSV_FILE" ]; then
    echo "Error: No CSV results file found in results/ directory"
    exit 1
fi

echo "Using results file: $CSV_FILE"

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

echo "Spark analysis completed at: $(date)"

# Show results
echo "Analysis results:"
ls -la analysis_results/

echo "Job completed successfully!"
