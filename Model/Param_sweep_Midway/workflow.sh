#!/bin/bash
"""
Complete ABM Parameter Sweep Workflow for Midway Cluster
This script orchestrates the entire process from parameter sweep to analysis
"""

set -e  # Exit on any error

echo "=========================================="
echo "ABM Parameter Sweep Workflow"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Configuration
SWEEP_JOB_NAME="abm-parameter-sweep"
ANALYSIS_JOB_NAME="abm-spark-analysis"
RESULTS_DIR="results"
ANALYSIS_DIR="analysis_results"

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p $ANALYSIS_DIR
mkdir -p logs

echo "Step 1: Submitting parameter sweep job..."
echo "----------------------------------------"

# Submit the parameter sweep job
SWEEP_JOB_ID=$(sbatch --parsable abm_sweep.sbatch)
echo "Parameter sweep job submitted with ID: $SWEEP_JOB_ID"

# Wait for the parameter sweep to complete
echo "Waiting for parameter sweep to complete..."
while [ $(squeue -j $SWEEP_JOB_ID -h | wc -l) -gt 0 ]; do
    sleep 30
    echo "  Still running... ($(date))"
done

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Parameter sweep completed successfully!"
else
    echo "Parameter sweep failed. Check logs."
    exit 1
fi

echo ""
echo "Step 2: Checking parameter sweep results..."
echo "-------------------------------------------"

# Check if results files were created
CSV_FILES=$(ls -1 $RESULTS_DIR/abm_sweep_results_*.csv 2>/dev/null | wc -l)
if [ $CSV_FILES -eq 0 ]; then
    echo "Error: No CSV results files found!"
    exit 1
fi

LATEST_CSV=$(ls -t $RESULTS_DIR/abm_sweep_results_*.csv | head -n 1)
echo "Found results file: $LATEST_CSV"

# Show basic file info
echo "File size: $(du -h $LATEST_CSV | cut -f1)"
echo "Number of rows: $(wc -l < $LATEST_CSV)"

echo ""
echo "Step 3: Submitting Spark analysis job..."
echo "----------------------------------------"

# Modify the analysis script to use the specific CSV file
sed "s|CSV_FILE=\$(ls -t results/abm_sweep_results_\*.csv | head -n 1)|CSV_FILE=\"$LATEST_CSV\"|" spark_analysis.sbatch > temp_spark_analysis.sbatch

# Submit the Spark analysis job
ANALYSIS_JOB_ID=$(sbatch --parsable temp_spark_analysis.sbatch)
echo "Spark analysis job submitted with ID: $ANALYSIS_JOB_ID"

# Wait for analysis to complete
echo "Waiting for Spark analysis to complete..."
while [ $(squeue -j $ANALYSIS_JOB_ID -h | wc -l) -gt 0 ]; do
    sleep 30
    echo "  Still running... ($(date))"
done

# Clean up temporary file
rm temp_spark_analysis.sbatch

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Spark analysis completed successfully!"
else
    echo "Spark analysis failed. Check logs."
    exit 1
fi

echo ""
echo "Step 4: Final results summary..."
echo "--------------------------------"

# Show what was created
echo "Parameter sweep results:"
ls -la $RESULTS_DIR/

echo ""
echo "Analysis results:"
ls -la $ANALYSIS_DIR/

echo ""
echo "Log files:"
ls -la *.out *.err 2>/dev/null || echo "No log files in current directory"

echo ""
echo "=========================================="
echo "Workflow completed successfully!"
echo "End time: $(date)"
echo "=========================================="

# Create a simple summary
TOTAL_SIMS=$(tail -n +2 "$LATEST_CSV" | wc -l)
echo ""
echo "SUMMARY:"
echo "  Total simulations run: $TOTAL_SIMS"
echo "  Results file: $LATEST_CSV"
echo "  Analysis directory: $ANALYSIS_DIR"

# Find the summary report
REPORT_FILE=$(ls -t $ANALYSIS_DIR/summary_report_*.md 2>/dev/null | head -n 1)
if [ -n "$REPORT_FILE" ]; then
    echo "  Summary report: $REPORT_FILE"
    echo ""
    echo "Key findings (first few lines of report):"
    head -n 20 "$REPORT_FILE"
fi
