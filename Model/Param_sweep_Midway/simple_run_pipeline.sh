#!/bin/bash
# Simplified pipeline that uses module python directly (no conda environment needed)

set -e  # Exit on any error

echo "============================================"
echo "ABM Simulation Pipeline for Midway (Simple)"
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

# Check if required packages are available
echo "Checking Python environment..."
module load python/anaconda-2022.05

# Test if required packages are available
python -c "import numpy, pandas, matplotlib, seaborn; print('Required packages available')" 2>/dev/null || {
    echo "ERROR: Required Python packages not available"
    echo "Try installing with:"
    echo "  pip install --user numpy pandas matplotlib seaborn"
    exit 1
}

echo "Python environment ready."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "============================================"

# Create simplified batch scripts that don't use conda
cat > simple_sweep.sbatch << 'EOF'
#!/bin/bash

#SBATCH --job-name=abm-param-sweep-simple
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

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Print job info
echo "ABM Parameter Sweep Job Started: $(date)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run the parameter sweep
python param_sweep.py \
    --min_informality 0.1 \
    --max_informality 0.8 \
    --informality_steps 8 \
    --n_seeds 100 \
    --n_processes 20 \
    --output_dir results

echo "Parameter sweep completed: $(date)"
EOF

cat > simple_analysis.sbatch << 'EOF'
#!/bin/bash

#SBATCH --job-name=abm-analysis-simple
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

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "ABM Analysis Job Started: $(date)"

# Find the most recent CSV results file
CSV_FILE=$(ls -t results/abm_sweep_results_*.csv 2>/dev/null | head -n 1)

if [ -z "$CSV_FILE" ]; then
    echo "ERROR: No CSV results file found"
    exit 1
fi

echo "Using results file: $CSV_FILE"

# Run quick analysis
python quick_analysis.py \
    --csv_file "$CSV_FILE" \
    --output_dir analysis_results

echo "Analysis completed: $(date)"
EOF

# Submit parameter sweep job
echo "Submitting parameter sweep job..."
SWEEP_JOB_ID=$(sbatch --parsable simple_sweep.sbatch)

if [ $? -eq 0 ]; then
    echo "Parameter sweep job submitted with ID: $SWEEP_JOB_ID"
else
    echo "ERROR: Failed to submit parameter sweep job"
    exit 1
fi

# Submit analysis job with dependency
echo "Submitting analysis job..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$SWEEP_JOB_ID simple_analysis.sbatch)

if [ $? -eq 0 ]; then
    echo "Analysis job submitted with ID: $ANALYSIS_JOB_ID"
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
echo ""
echo "Check logs:"
echo "  tail -f logs/abm_sweep_${SWEEP_JOB_ID}.out"
echo "  tail -f logs/analysis_${ANALYSIS_JOB_ID}.out"
echo "============================================"
