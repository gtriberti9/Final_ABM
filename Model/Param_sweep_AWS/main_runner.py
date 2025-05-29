#!/usr/bin/env python3
"""
Main controller for ABM Parameter Sweep
Handles both local and AWS execution modes
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_local_sweep(config):
    """Run parameter sweep locally"""
    from parameter_sweep_runner import ParameterSweepRunner
    from analysis_visualization import analyze_sweep_results
    
    logger.info("Starting local parameter sweep...")
    
    # Initialize runner
    runner = ParameterSweepRunner(
        s3_bucket=config.get('s3_bucket'),
        local_results_dir=config.get('local_results_dir', 'sweep_results')
    )
    
    # Run sweep
    results_df = runner.run_parameter_sweep(
        informality_rates=config.get('informality_rates'),
        seeds=config.get('seeds'),
        n_processes=config.get('n_processes'),
        **config.get('model_params', {})
    )
    
    # Save detailed results
    runner.save_detailed_results(results_df)
    
    # Generate analysis
    if config.get('auto_analyze', True):
        logger.info("Starting analysis...")
        
        # Find the most recent results file
        import glob
        result_files = glob.glob("sweep_results/parameter_sweep_results_*.csv")
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            logger.info(f"Analyzing results from: {latest_file}")
            
            try:
                # Try the comprehensive analysis first
                from analysis_visualization import analyze_sweep_results
                analyze_sweep_results(latest_file)
            except Exception as e:
                logger.warning(f"Comprehensive analysis failed: {str(e)}")
                logger.info("Falling back to simple analysis...")
                
                try:
                    # Fall back to simple analysis
                    from simple_analysis import load_and_clean_results, create_summary_analysis, print_key_findings
                    df = load_and_clean_results(latest_file)
                    create_summary_analysis(df)
                    print_key_findings(df)
                except Exception as e2:
                    logger.error(f"Simple analysis also failed: {str(e2)}")
                    logger.info("You can manually analyze the results CSV file.")
        else:
            logger.warning("No results files found for analysis")
    
    return results_df

def run_aws_sweep(config):
    """Run parameter sweep on AWS"""
    from Param_sweep_AWS.aws_test_deployment import deploy_parameter_sweep
    
    logger.info("Starting AWS parameter sweep deployment...")
    
    # Deploy to AWS
    deployment = deploy_parameter_sweep(config['aws_config'])
    
    if deployment:
        # Monitor and collect results
        logger.info("Monitoring AWS deployment...")
        status = deployment.monitor_instances()
        
        logger.info("Collecting results from AWS...")
        result_files = deployment.collect_results()
        
        # Analyze results if requested
        if config.get('auto_analyze', True) and result_files:
            from analysis_visualization import analyze_sweep_results
            # Find CSV results file
            csv_files = [f for f in result_files if f.endswith('.csv')]
            if csv_files:
                analyze_sweep_results(csv_files[0])
        
        # Cleanup
        if config.get('auto_cleanup', True):
            deployment.cleanup_deployment()
        
        return result_files
    
    return None

def create_default_config():
    """Create default configuration"""
    return {
        'mode': 'local',  # 'local' or 'aws'
        'informality_rates': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'seeds': list(range(100)),
        'n_processes': None,  # Auto-detect for local
        'auto_analyze': True,
        'auto_cleanup': True,
        
        # Model parameters
        'model_params': {
            'n_firms': 50,
            'n_consumers': 200,
            'n_banks': 5,
            'max_steps': 200,
            'inflation_target': 0.02,
            'initial_policy_rate': 0.03,
            'current_inflation': 0.02
        },
        
        # Local execution
        'local_results_dir': 'sweep_results',
        's3_bucket': None,  # Optional for local mode
        
        # AWS configuration
        'aws_config': {
            'region_name': 'us-east-1',
            'instance_type': 'c5.2xlarge',
            'num_instances': 4,
            's3_bucket': 'your-abm-results-bucket',
            'key_pair_name': None,
            'security_group_id': None,
            'subnet_id': None,
            'estimated_runtime_hours': 2.0,
            'auto_confirm': False
        }
    }

def load_config(config_file):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return create_default_config()
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        return create_default_config()

def save_config(config, config_file):
    """Save configuration to file"""
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def validate_config(config):
    """Validate configuration parameters"""
    errors = []
    
    # Check mode
    if config.get('mode') not in ['local', 'aws']:
        errors.append("Mode must be 'local' or 'aws'")
    
    # Check informality rates
    informality_rates = config.get('informality_rates', [])
    if not informality_rates or not all(0 <= rate <= 1 for rate in informality_rates):
        errors.append("Informality rates must be between 0 and 1")
    
    # Check seeds
    seeds = config.get('seeds', [])
    if not seeds or not all(isinstance(s, int) for s in seeds):
        errors.append("Seeds must be a list of integers")
    
    # AWS-specific validation
    if config.get('mode') == 'aws':
        aws_config = config.get('aws_config', {})
        if not aws_config.get('s3_bucket'):
            errors.append("S3 bucket is required for AWS mode")
    
    # Model parameters validation
    model_params = config.get('model_params', {})
    required_params = ['n_firms', 'n_consumers', 'n_banks', 'max_steps']
    for param in required_params:
        if param not in model_params or not isinstance(model_params[param], int):
            errors.append(f"Model parameter '{param}' must be an integer")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

def print_config_summary(config):
    """Print configuration summary"""
    print("\n" + "="*50)
    print("PARAMETER SWEEP CONFIGURATION")
    print("="*50)
    print(f"Mode: {config['mode'].upper()}")
    print(f"Informality rates: {len(config['informality_rates'])} values ({min(config['informality_rates']):.1f} to {max(config['informality_rates']):.1f})")
    print(f"Random seeds: {len(config['seeds'])} seeds")
    print(f"Total simulations: {len(config['informality_rates']) * len(config['seeds'])}")
    
    print("\nModel Parameters:")
    for key, value in config['model_params'].items():
        print(f"  {key}: {value}")
    
    if config['mode'] == 'aws':
        aws_config = config['aws_config']
        print(f"\nAWS Configuration:")
        print(f"  Instance type: {aws_config['instance_type']}")
        print(f"  Number of instances: {aws_config['num_instances']}")
        print(f"  S3 bucket: {aws_config['s3_bucket']}")
        print(f"  Estimated runtime: {aws_config['estimated_runtime_hours']} hours")
    
    print(f"\nAuto-analyze results: {config['auto_analyze']}")
    print("="*50)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ABM Parameter Sweep Runner')
    parser.add_argument('--config', '-c', default='sweep_config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', choices=['local', 'aws'],
                       help='Execution mode (overrides config)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--validate-only', action='store_true',
                       help='Validate configuration and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal parameters')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        save_config(config, args.config)
        print(f"Default configuration created: {args.config}")
        print("Edit the configuration file and run again to execute the sweep.")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override mode if specified
    if args.mode:
        config['mode'] = args.mode
    
    # Quick test mode
    if args.quick_test:
        logger.info("Running in quick test mode...")
        config['informality_rates'] = [0.1, 0.3, 0.5]
        config['seeds'] = list(range(5))
        config['model_params']['max_steps'] = 50
        if config['mode'] == 'aws':
            config['aws_config']['num_instances'] = 1
            config['aws_config']['auto_confirm'] = True
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration is valid.")
        return
    
    # Print configuration summary
    print_config_summary(config)
    
    # Dry run
    if args.dry_run:
        print("\nDRY RUN - No actual execution performed")
        if config['mode'] == 'local':
            print(f"Would run {len(config['informality_rates']) * len(config['seeds'])} simulations locally")
            if config.get('n_processes'):
                print(f"Using {config['n_processes']} processes")
        else:
            print(f"Would deploy to AWS with {config['aws_config']['num_instances']} instances")
        return
    
    # Confirm execution (unless auto-confirmed)
    if not config.get('auto_confirm', False) and not args.quick_test:
        total_sims = len(config['informality_rates']) * len(config['seeds'])
        if config['mode'] == 'aws':
            from Param_sweep_AWS.aws_test_deployment import AWSParameterSweepDeployment
            deployment = AWSParameterSweepDeployment()
            cost_estimate = deployment.estimate_costs(
                config['aws_config']['num_instances'],
                config['aws_config']['estimated_runtime_hours']
            )
            confirm_msg = f"\nRun {total_sims} simulations on AWS? Estimated cost: ${cost_estimate['total_estimated_cost']:.2f}"
        else:
            confirm_msg = f"\nRun {total_sims} simulations locally?"
        
        if input(f"{confirm_msg} (y/n): ").lower() != 'y':
            print("Execution cancelled.")
            return
    
    # Execute parameter sweep
    start_time = time.time()
    
    try:
        if config['mode'] == 'local':
            results = run_local_sweep(config)
            logger.info(f"Local sweep completed with {len(results)} results")
        else:
            results = run_aws_sweep(config)
            logger.info(f"AWS sweep completed")
        
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time/60:.1f} minutes")
        
        # Save final config for reference
        config['execution_time'] = total_time
        config['execution_timestamp'] = datetime.now().isoformat()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_config(config, f"executed_config_{timestamp}.json")
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)

# Convenience functions for different execution modes
def run_quick_local_test():
    """Run a quick local test with minimal parameters"""
    config = create_default_config()
    config['informality_rates'] = [0.1, 0.5, 0.8]
    config['seeds'] = list(range(10))
    config['model_params']['max_steps'] = 100
    
    print("Running quick local test...")
    results = run_local_sweep(config)
    print(f"Test completed with {len(results)} results")
    return results

def run_full_local_sweep():
    """Run full local parameter sweep"""
    config = create_default_config()
    print("Running full local parameter sweep...")
    results = run_local_sweep(config)
    print(f"Full sweep completed with {len(results)} results")
    return results

if __name__ == "__main__":
    main()