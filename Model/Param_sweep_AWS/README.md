# ABM Parameter Sweep Setup and Usage Guide

This guide will help you set up and run parallelized parameter sweeps for your Agent-Based Model (ABM) both locally and on AWS.

## Quick Start

### 1. Local Execution (Simplest)

```bash
# Create default configuration
python main_runner.py --create-config

# Run quick test (5 seeds, 3 informality rates, 50 steps)
python main_runner.py --quick-test

# Run full local sweep
python main_runner.py --mode local
```

### 2. AWS Execution (Most Scalable)

```bash
# Edit configuration for AWS
python main_runner.py --create-config
# Edit sweep_config.json to set your AWS settings
python main_runner.py --mode aws
```

## File Structure

```
your_project/
├── model.py                    # Your existing ABM model
├── agents.py                   # Your existing agent definitions
├── parameter_sweep_runner.py   # Core parameter sweep engine
├── analysis_visualization.py   # Results analysis and visualization
├── aws_deployment.py          # AWS deployment automation
├── main_runner.py             # Main execution controller
├── sweep_config.json          # Configuration file
└── sweep_results/             # Local results directory
```

## Configuration

The system uses a JSON configuration file (`sweep_config.json`) to control all parameters:

```json
{
  "mode": "local",
  "informality_rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "seeds": [0, 1, 2, ..., 99],
  "auto_analyze": true,
  "model_params": {
    "n_firms": 50,
    "n_consumers": 200,
    "n_banks": 5,
    "max_steps": 200,
    "inflation_target": 0.02,
    "initial_policy_rate": 0.03,
    "current_inflation": 0.02
  },
  "aws_config": {
    "instance_type": "c5.2xlarge",
    "num_instances": 4,
    "s3_bucket": "your-bucket-name",
    "estimated_runtime_hours": 2.0
  }
}
```

## Local Execution

### Prerequisites
- Python 3.8+
- Required packages (install via pip):
  ```bash
  pip install numpy pandas matplotlib scipy mesa boto3 plotly seaborn tqdm
  ```

### Running Locally

```bash
# Validate configuration
python main_runner.py --validate-only

# Dry run (see what would execute)
python main_runner.py --dry-run

# Execute with default settings
python main_runner.py

# Execute with custom config
python main_runner.py --config my_config.json
```

### Local Performance
- **8 informality rates × 100 seeds = 800 simulations**
- **Estimated time**: 30-60 minutes on modern laptop (8 cores)
- **Memory usage**: ~2-4 GB RAM
- **CPU usage**: All available cores (configurable)

## AWS Execution

### Prerequisites

1. **AWS Account Setup**:
   ```bash
   # Install AWS CLI
   pip install awscli
   
   # Configure credentials
   aws configure
   ```

2. **Create S3 Bucket**:
   ```bash
   aws s3 mb s3://your-abm-results-bucket
   ```

3. **Create IAM Role** (for EC2 instances):
   - Role name: `EC2-S3-Access-Role`
   - Attach policies: `AmazonS3FullAccess`, `CloudWatchLogsFullAccess`

4. **Security Group** (optional but recommended):
   ```bash
   # Create security group allowing SSH access
   aws ec2 create-security-group --group-name abm-sweep-sg --description "ABM Parameter Sweep"
   ```

### AWS Configuration

Edit your `sweep_config.json`:

```json
{
  "mode": "aws",
  "aws_config": {
    "region_name": "us-east-1",
    "instance_type": "c5.2xlarge",
    "num_instances": 4,
    "s3_bucket": "your-abm-results-bucket",
    "key_pair_name": "your-key-pair",
    "security_group_id": "sg-xxxxxxxxx",
    "estimated_runtime_hours": 2.0,
    "auto_confirm": false
  }
}
```

### Running on AWS

```bash
# Deploy to AWS
python main_runner.py --mode aws

# Monitor progress (in separate terminal)
aws ec2 describe-instances --filters "Name=tag:Project,Values=ABM-Parameter-Sweep" --query "Reservations[].Instances[].State.Name"
```

### AWS Performance & Costs
- **Instance type**: c5.2xlarge (8 vCPUs, 16 GB RAM)
- **4 instances × 2 hours**: ~$2.72 total cost
- **800 simulations**: ~30-60 minutes total time
- **Parallel efficiency**: ~4x speedup with 4 instances

## Results Analysis

The system automatically generates comprehensive analysis:

### Outputs Generated
1. **CSV Results**: `parameter_sweep_results_TIMESTAMP.csv`
2. **Detailed Data**: `detailed_results_TIMESTAMP.pkl`
3. **Visualizations**: `parameter_sweep_results.png`
4. **Interactive Dashboard**: `interactive_dashboard.html`
5. **Statistical Analysis**: `complete_analysis_*.csv/json`

### Key Metrics Tracked
- Final inflation rate vs. informality
- Credit access gap by sector
- Production levels (formal vs informal)
- Policy rate adjustments
- Convergence rates
- Banking system metrics

### Analysis Features
- **Summary statistics** by informality rate
- **Statistical tests** (ANOVA, regression)
- **Interactive visualizations** with Plotly
- **Correlation analysis**
- **Policy insights** and recommendations

## Usage Examples

### Example 1: Quick Local Test
```python
# In Python
from main_runner import run_quick_local_test
results = run_quick_local_test()
```

### Example 2: Custom Parameter Range
```python
from parameter_sweep_runner import ParameterSweepRunner

runner = ParameterSweepRunner()
results = runner.run_parameter_sweep(
    informality_rates=[0.2, 0.4, 0.6],
    seeds=range(50),
    n_firms=30,
    n_consumers=150,
    max_steps=150
)
```

### Example 3: AWS Deployment with Custom Settings
```python
from aws_deployment import deploy_parameter_sweep

config = {
    'instance_type': 'c5.4xlarge',  # Larger instances
    'num_instances': 8,             # More parallelism
    's3_bucket': 'my-results-bucket',
    'estimated_runtime_hours': 1.5
}

deployment = deploy_parameter_sweep({'aws_config': config})
```

## Troubleshooting

### Common Issues

1. **Memory errors locally**:
   - Reduce number of parallel processes
   - Reduce model size (fewer agents)
   - Increase system swap space

2. **AWS permissions**:
   - Ensure IAM role has S3 access
   - Check security group allows outbound internet
   - Verify AWS credentials are configured

3. **Long runtime**:
   - Enable early convergence stopping
   - Reduce max_steps
   - Use more AWS instances

4. **Results not found**:
   - Check S3 bucket permissions
   - Verify deployment completed successfully
   - Look for error logs in EC2 console

### Performance Optimization

1. **Local Optimization**:
   ```python
   # Use fewer processes if memory constrained
   n_processes = 4  # Instead of auto-detect
   
   # Reduce model complexity
   n_firms = 30      # Instead of 50
   n_consumers = 100 # Instead of 200
   max_steps = 150   # Instead of 200
   ```

2. **AWS Optimization**:
   ```python
   # Use compute-optimized instances for CPU-intensive work
   instance_type = 'c5.4xlarge'  # 16 vCPUs
   
   # Distribute work across more instances
   num_instances = 8
   
   # Use spot instances for cost savings (advanced)
   # Note: Requires additional configuration
   ```

## Monitoring and Logs

### Local Monitoring
- Console output shows progress in real-time
- Check CPU usage with system monitor
- Results saved incrementally to disk

### AWS Monitoring
- Use EC2 console to check instance status
- CloudWatch logs for detailed execution logs
- S3 console to monitor result uploads

### Log Locations
- **Local**: Console output and log files in current directory
- **AWS**: CloudWatch logs group `/aws/ec2/parameter-sweep`
- **Results**: S3 bucket under `abm_sweep_results/` prefix

## Best Practices

1. **Start Small**: Always run quick tests before full sweeps
2. **Version Control**: Save configurations with timestamps
3. **Cost Management**: Set AWS billing alerts
4. **Backup Results**: Download results from S3 promptly
5. **Resource Cleanup**: Always terminate AWS instances when done

## Support and Extension

The system is designed to be easily extensible:

- **Add new parameters**: Modify `generate_parameter_combinations()`
- **Custom analysis**: Extend `ParameterSweepAnalyzer` class
- **Different cloud providers**: Create new deployment classes
- **Additional metrics**: Modify model data collection

For issues or enhancements, check the error logs and configuration validation output.