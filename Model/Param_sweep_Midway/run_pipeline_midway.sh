#!/bin/bash
# Complete pipeline to run ABM parameter sweep and analysis on Midway

set -e  # Exit on any error

echo "============================================"
echo "ABM Simulation Pipeline for Midway"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "model.py" ] || [ ! -f "param_sweep.py" ]; then
    echo "ERROR: Required Python files not found in current directory"
    echo "Make sure you're in the directory containing:"
    echo "  - model.py"
    echo "  - param_sweep.py" 
    echo "  - quick_analysis.py"
    echo "  - spark_analysis.py (optional)"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p {results,analysis_results,logs}

# Load modules and activate environment
echo "Loading modules and environment..."
module load python/anaconda-2022.05

# Check if conda environment exists
if conda info --envs | grep -q "abm_env"; then
    echo "Activating existing abm_env environment..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate abm_env
else
    echo "ERROR: abm_env conda environment not found"
    echo "Please run the setup script first:"
    echo "  bash setup_midway.sh"
    exit 1
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "Environment setup complete."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "============================================"

# Submit parameter sweep job
echo "Submitting parameter sweep job..."
SWEEP_JOB_ID=$(sbatch --parsable run_sweep.sbatch)

if [ $? -eq 0 ]; then
    echo "Parameter sweep job submitted with ID: $SWEEP_JOB_ID"
else
    echo "ERROR: Failed to submit parameter sweep job"
    exit 1
fi

echo "============================================"

# Submit analysis job with dependency on sweep job
echo "Submitting analysis job (depends on sweep completion)..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$SWEEP_JOB_ID run_analysis.sbatch)

if [ $? -eq 0 ]; then
    echo "Analysis job submitted with ID: $ANALYSIS_JOB_ID"
    echo "Analysis will start after sweep job $SWEEP_JOB_ID completes successfully"
else
    echo "ERROR: Failed to submit analysis job"
    exit 1
fi

echo "============================================"
echo "Pipeline submitted successfully!"
echo ""
echo "Job Summary:"
echo "  Parameter Sweep Job ID: $SWEEP_JOB_ID"
echo "  Analysis Job ID: $ANALYSIS_JOB_ID"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo "  squeue -j $SWEEP_JOB_ID,$ANALYSIS_JOB_ID"
echo ""
echo "Check logs in:"
echo "  logs/abm_sweep_${SWEEP_JOB_ID}.out"
echo "  logs/analysis_${ANALYSIS_JOB_ID}.out"
echo ""
echo "Results will be in:"
echo "  results/        (simulation data)"
echo "  analysis_results/ (plots and analysis)"
echo "============================================"

# Create a status checking script
cat > check_status.sh << 'EOF'
#!/bin/bash
echo "ABM Pipeline Status Check"
echo "========================"
echo "Current time: $(date)"
echo ""

# Check job status
echo "Job Status:"
squeue -u $USER --format="%.10i %.12j %.8T %.10M %.6D %R" || echo "No jobs in queue"
echo ""

# Check for results
if [ -d "results" ]; then
    echo "Simulation Results:"
    echo "  Files in results/: $(ls results/ 2>/dev/null | wc -l)"
    LATEST_CSV=$(ls -t results/abm_sweep_results_*.csv 2>/dev/null | head -n 1)
    if [ ! -z "$LATEST_CSV" ]; then
        echo "  Latest CSV: $(basename "$LATEST_CSV")"
        echo "  CSV size: $(du -h "$LATEST_CSV" | cut -f1)"
        echo "  CSV rows: $(wc -l < "$LATEST_CSV")"
    fi
    echo ""
fi

if [ -d "analysis_results" ]; then
    echo "Analysis Results:"
    echo "  Files in analysis_results/: $(ls analysis_results/ 2>/dev/null | wc -l)"
    echo "  Plots generated: $(find analysis_results/ -name "*.png" 2>/dev/null | wc -l)"
    echo ""
fi

echo "Recent log entries:"
echo "-------------------"
for log in logs/*.out; do
    if [ -f "$log" ]; then
        echo "From $(basename "$log"):"
        tail -n 3 "$log" 2>/dev/null
        echo ""
    fi
done
EOF

chmod +x check_status.sh

echo "Created check_status.sh script to monitor progress"
echo "Run './check_status.sh' to check pipeline status"
echo ""
echo "Pipeline setup complete!"
