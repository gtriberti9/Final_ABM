# ABM Parameter Sweep on Midway Cluster

This repository contains scripts for running parallelized parameter sweeps of the Agent-Based Monetary Policy Model with informality dynamics, followed by scalable analysis using Apache Spark.

## Files Overview

### Core Model Files
- `model.py` - Main ABM model implementation
- `agents.py` - Agent classes (CentralBank, CommercialBank, Firm, Consumer)
- `app.py` - Solara interactive interface (not needed for parameter sweep)

### Parameter Sweep
- `param_sweep.py` - Main parameter sweep runner with multiprocessing
- `abm_sweep.sbatch` - SLURM batch script for running parameter sweep on Midway

### Analysis
- `spark_analysis.py` - Comprehensive Spark-based analysis
- `spark_analysis.sbatch` - SLURM batch script for Spark analysis
- `quick_analysis.py` - Quick pandas-based analysis (alternative)

### Workflow
- `workflow.sh` - Complete workflow orchestrator
- `README.md` - This file

## Quick Start

### Option 1: Run Complete Workflow (Recommended)
```bash
# Make workflow executable
chmod +x workflow.sh

# Run complete workflow (parameter sweep + Spark analysis)
./workflow.sh
```

### Option 2: Step-by-Step Execution

#### Step 1: Parameter Sweep
```bash
# Submit parameter sweep job
sbatch abm_sweep.sbatch

# Monitor job
squeue -u $USER

# Check results
ls -la results/
```

#### Step 2: Analysis
```bash
# After parameter sweep completes, run Spark analysis
sbatch spark_analysis.sbatch

# Or run quick analysis locally
python quick_analysis.py --csv_file results/abm_sweep_results_TIMESTAMP.csv
```

## Configuration

### Parameter Sweep Configuration
Edit `abm_sweep.sbatch` or command line arguments in `param_sweep.py`:

```bash
python param_sweep.py \
    --min_informality 0.1 \        # Minimum informality rate
    --max_informality 0.8 \        # Maximum informality rate  
    --informality_steps 8 \        # Number of steps between min/max
    --n_seeds 100 \                # Random seeds per parameter set
    --n_processes 20 \             # Parallel processes
    --output_dir results           # Output directory
```

### SLURM Configuration
Adjust resource allocation in the `.sbatch` files:

```bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=20                # Number of tasks (= n_processes)
#SBATCH --mem=40G                  # Memory allocation
#SBATCH --time=04:00:00           # Time limit (HH:MM:SS)
#SBATCH --partition=caslake        # Partition name
#SBATCH --account=macs30123        # Account name
```

## Parameter Sweep Details

### Parameters Tested
- **Informality Rate**: 0.1 to 0.8 in 8 steps (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
- **Random Seeds**: 100 per informality rate
- **Total Simulations**: 8 × 100 = 800 simulations

### Model Configuration
- Firms: 50
- Consumers: 200  
- Commercial Banks: 5
- Inflation Target: 2%
- Initial Policy Rate: 3%
- Max Steps: 200
- Convergence Criterion: Inflation stable within [1%, 3%] for 36 consecutive periods

### Output Files
- `abm_sweep_results_TIMESTAMP.csv` - Main results CSV
- `abm_sweep_results_full_TIMESTAMP.pkl` - Full results with time series data
- `sweep_metadata_TIMESTAMP.json` - Metadata about the sweep

## Analysis Features

### Spark Analysis (`spark_analysis.py`)
- **Scalable Processing**: Handles large datasets efficiently
- **Comprehensive Statistics**: Overall, by informality rate, by regime
- **Correlation Analysis**: Between key variables
- **Convergence Analysis**: Patterns and timing
- **Economic Outcomes**: Production, consumption, credit metrics
- **Visualizations**: 9-panel analysis plots, correlation heatmaps
- **Summary Report**: Markdown report with key findings

### Quick Analysis (`quick_analysis.py`)
- **Fast Processing**: Pandas-based for smaller datasets
- **Key Metrics**: Convergence, inflation, credit gaps
- **Basic Visualizations**: Trends and distributions
- **JSON Export**: Results in structured format

## Results Structure

```
results/
├── abm_sweep_results_TIMESTAMP.csv      # Main results
├── abm_sweep_results_full_TIMESTAMP.pkl  # Full data with time series
└── sweep_metadata_TIMESTAMP.json        # Sweep configuration

analysis_results/
├── analysis_results_TIMESTAMP.json      # Detailed analysis results
├── abm_analysis_main_TIMESTAMP.png      # Main analysis plots
├── correlation_matrix_TIMESTAMP.png     # Correlation heatmap
├── distribution_analysis_TIMESTAMP.png  # Distribution plots
└── summary_report_TIMESTAMP.md          # Comprehensive report

quick_analysis/
├── quick_analysis_TIMESTAMP.png         # Quick analysis plots
├── distributions_TIMESTAMP.png          # Distribution plots
├── quick_analysis_results_TIMESTAMP.json # JSON results
└── summary_by_informality_TIMESTAMP.csv  # Summary table
```

## Key Metrics Analyzed

### Macroeconomic Indicators
- Final inflation rate vs. 2% target
- Policy rate levels and volatility
- Output gap dynamics
- Economic stability measures

### Sectoral Analysis
- Formal vs. informal production shares
- Credit access gaps between sectors
- Price differentials
- Agent distribution by formality

### Banking System
- Lending rates and spreads
- Credit allocation (formal vs. informal)
- Bank-level heterogeneity

### Policy Effectiveness
- Convergence rates and timing
- Distance from inflation target
- Volatility measures
- Monetary transmission strength

## Expected Findings

Based on the model structure, you should observe:

1. **Reduced Policy Effectiveness**: Higher informality rates should correlate with:
   - Lower convergence rates
   - Higher inflation volatility
   - Weaker monetary transmission

2. **Credit Market Segmentation**: 
   - Widening credit access gaps
   - Lower formal credit ratios
   - Higher informal lending rates

3. **Sectoral Imbalances**:
   - Productivity gaps between sectors
   - Price level differences
   - Unequal growth patterns

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `n_processes` or increase `--mem` in SLURM script
2. **Time Limits**: Increase `--time` or reduce `n_seeds`
3. **Missing Results**: Check `.err` files for error messages
4. **Spark Issues**: Ensure proper module loading and jar files

### Monitoring Jobs
```bash
# Check job status
squeue -u $USER

# Check job details  
scontrol show job JOB_ID

# Check resource usage
sacct -j JOB_ID --format=JobID,JobName,MaxRSS,Elapsed

# View output
tail -f abm_sweep_JOB_ID.out
```

### Performance Optimization

1. **Parallel Processing**: Match `n_processes` to `--ntasks` in SLURM
2. **Memory Management**: Monitor memory usage and adjust accordingly
3. **I/O Optimization**: Use local scratch space for temporary files
4. **Spark Tuning**: Adjust executor memory and cores based on data size

## Advanced Usage

### Custom Parameter Ranges
Modify the parameter sweep by editing `param_sweep.py` or command line arguments:

```python
# Example: Test different economic scenarios
informality_rates = [0.1, 0.2, 0.4, 0.6, 0.8]  # Custom rates
n_seeds = 50  # Fewer seeds for testing
```

### Extended Analysis
Add more metrics to the analysis by modifying `run_single_simulation()` in `param_sweep.py`:

```python
# Add custom metrics
'custom_metric': some_calculation(model),
'sector_inequality': gini_coefficient(agent_wealths),
```

### Batch Processing Multiple Scenarios
Create multiple parameter sweep configurations:

```bash
# Run different inflation targets
python param_sweep.py --inflation_target 0.015 --output_dir results_low_target
python param_sweep.py --inflation_target 0.025 --output_dir results_high_target
```

## References

- Mesa ABM Framework: https://mesa.readthedocs.io/
- Apache Spark: https://spark.apache.org/docs/latest/
- Midway Cluster Documentation: https://rcc.uchicago.edu/docs/
- SLURM Documentation: https://slurm.schedmd.com/documentation.html

## Support

For issues specific to:
- **Model Implementation**: Check `model.py` and `agents.py` 
- **Parameter Sweep**: Review `param_sweep.py` and job logs
- **Spark Analysis**: Check Spark logs and ensure proper setup
- **Midway Cluster**: Consult RCC documentation or support

## License

This code is provided for academic research purposes. Please cite appropriately if used in publications.