import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import json
import time
import os
from typing import Dict, List, Tuple
import boto3
from botocore.exceptions import NoCredentialsError
import pickle
import logging
from datetime import datetime

# Import your model components
from model import MonetaryPolicyModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParameterSweepRunner:
    """
    Parallel parameter sweep runner for ABM simulations with AWS integration
    """
    
    def __init__(self, 
                 s3_bucket=None,
                 s3_prefix="abm_sweep_results",
                 local_results_dir="sweep_results"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.local_results_dir = local_results_dir
        self.s3_client = None
        
        # Create local results directory
        os.makedirs(local_results_dir, exist_ok=True)
        
        # Initialize S3 client if bucket provided
        if s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"S3 client initialized for bucket: {s3_bucket}")
            except NoCredentialsError:
                logger.warning("AWS credentials not found. Results will only be saved locally.")
                self.s3_client = None
    
    def run_single_simulation(self, params: Dict) -> Dict:
        """
        Run a single simulation with given parameters
        """
        try:
            # Extract parameters
            informality_rate = params['informality_rate']
            seed = params['seed']
            n_firms = params.get('n_firms', 50)
            n_consumers = params.get('n_consumers', 200)
            n_banks = params.get('n_banks', 5)
            max_steps = params.get('max_steps', 200)
            inflation_target = params.get('inflation_target', 0.02)
            initial_policy_rate = params.get('initial_policy_rate', 0.03)
            current_inflation = params.get('current_inflation', 0.02)
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create and run model
            model = MonetaryPolicyModel(
                n_firms=n_firms,
                n_consumers=n_consumers,
                n_commercial_banks=n_banks,
                inflation_target=inflation_target,
                initial_policy_rate=initial_policy_rate,
                informality_rate=informality_rate,
                current_inflation=current_inflation
            )
            
            # Run simulation
            start_time = time.time()
            for step in range(max_steps):
                model.step()
                
                # Early stopping if inflation stabilizes
                if step > 50 and len(model.inflation_history) >= 36:
                    last_36 = model.inflation_history[-36:]
                    if all(0.01 <= val <= 0.03 for val in last_36):
                        logger.info(f"Early convergence at step {step} for seed {seed}, informality {informality_rate}")
                        break
            
            run_time = time.time() - start_time
            
            # Collect final results
            final_results = {
                'parameters': params,
                'run_time': run_time,
                'final_step': model.time,
                'converged': len(model.inflation_history) >= 36 and all(0.01 <= val <= 0.03 for val in model.inflation_history[-36:]),
                
                # Final state metrics
                'final_inflation': model.current_inflation,
                'final_policy_rate': model.policy_rate,
                'final_output_gap': model.output_gap,
                'avg_lending_rate': np.mean([bank.lending_rate for bank in model.commercial_banks]),
                
                # Informality metrics
                'informal_firm_pct': len([f for f in model.firms if f.is_informal]) / len(model.firms) * 100,
                'informal_consumer_pct': len([c for c in model.consumers if c.is_informal]) / len(model.consumers) * 100,
                'credit_access_gap': model._calculate_credit_gap(),
                
                # Sector analysis
                'sector_analysis': model.get_sector_analysis(),
                
                # Time series (last 50 points to save space)
                'inflation_history': model.inflation_history[-50:],
                'policy_rate_history': model.policy_rate_history[-50:],
                'output_gap_history': model.output_gap_history[-50:],
                
                # Banking metrics
                'total_formal_loans': sum([bank.formal_loans for bank in model.commercial_banks]),
                'total_informal_loans': sum([bank.informal_loans for bank in model.commercial_banks]),
                'avg_credit_tightness': np.mean([bank.credit_tightness for bank in model.commercial_banks]),
                
                # Aggregate metrics
                'total_production': sum([firm.production for firm in model.firms]),
                'total_consumption': sum([consumer.consumption for consumer in model.consumers]),
                'formal_production': sum([f.production for f in model.firms if not f.is_informal]),
                'informal_production': sum([f.production for f in model.firms if f.is_informal]),
            }
            
            logger.info(f"Completed simulation: seed={seed}, informality={informality_rate:.1f}, steps={model.time}, time={run_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in simulation seed={seed}, informality={informality_rate}: {str(e)}")
            return {
                'parameters': params,
                'error': str(e),
                'run_time': 0,
                'converged': False
            }

    def generate_parameter_combinations(self, 
                                      informality_rates=None, 
                                      seeds=None,
                                      **fixed_params) -> List[Dict]:
        """
        Generate all parameter combinations for the sweep
        """
        if informality_rates is None:
            informality_rates = np.arange(0.1, 0.9, 0.1)  # 0.1 to 0.8 by 0.1
        
        if seeds is None:
            seeds = range(100)  # 100 random seeds
        
        combinations = []
        for informality_rate in informality_rates:
            for seed in seeds:
                params = {
                    'informality_rate': informality_rate,
                    'seed': seed,
                    **fixed_params
                }
                combinations.append(params)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations

    def run_parameter_sweep(self, 
                          informality_rates=None,
                          seeds=None,
                          n_processes=None,
                          **fixed_params) -> pd.DataFrame:
        """
        Run the full parameter sweep in parallel
        """
        # Determine number of processes
        if n_processes is None:
            n_processes = min(mp.cpu_count(), 8)  # Use up to 8 cores
        
        logger.info(f"Starting parameter sweep with {n_processes} processes")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(
            informality_rates, seeds, **fixed_params
        )
        
        # Run simulations in parallel
        start_time = time.time()
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(self.run_single_simulation, param_combinations)
        
        total_time = time.time() - start_time
        logger.info(f"Completed {len(results)} simulations in {total_time:.2f} seconds")
        
        # Convert results to DataFrame
        results_df = pd.json_normalize(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parameter_sweep_results_{timestamp}.csv"
        
        # Save locally
        local_path = os.path.join(self.local_results_dir, filename)
        results_df.to_csv(local_path, index=False)
        logger.info(f"Results saved locally to: {local_path}")
        
        # Save to S3 if available
        if self.s3_client and self.s3_bucket:
            try:
                s3_key = f"{self.s3_prefix}/{filename}"
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                logger.info(f"Results uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload to S3: {str(e)}")
        
        return results_df

    def save_detailed_results(self, results_df: pd.DataFrame, include_time_series=True):
        """
        Save detailed results including time series data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if include_time_series:
            # Save as pickle to preserve complex data structures
            pickle_filename = f"detailed_results_{timestamp}.pkl"
            pickle_path = os.path.join(self.local_results_dir, pickle_filename)
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(results_df, f)
            
            logger.info(f"Detailed results saved to: {pickle_path}")
            
            # Upload to S3 if available
            if self.s3_client and self.s3_bucket:
                try:
                    s3_key = f"{self.s3_prefix}/{pickle_filename}"
                    self.s3_client.upload_file(pickle_path, self.s3_bucket, s3_key)
                    logger.info(f"Detailed results uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload detailed results to S3: {str(e)}")


def run_sweep_wrapper(args):
    """Wrapper function for running simulation - needed for multiprocessing"""
    return ParameterSweepRunner().run_single_simulation(args)


# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'informality_rates': np.arange(0.1, 0.9, 0.1),  # 0.1 to 0.8 by 0.1
        'seeds': range(100),  # 100 random seeds
        'n_processes': None,  # Auto-detect
        's3_bucket': 'your-s3-bucket-name',  # Replace with your S3 bucket
        's3_prefix': 'abm_sweep_results',
        
        # Fixed model parameters
        'n_firms': 50,
        'n_consumers': 200,
        'n_banks': 5,
        'max_steps': 200,
        'inflation_target': 0.02,
        'initial_policy_rate': 0.03,
        'current_inflation': 0.12
    }
    
    # Initialize runner
    runner = ParameterSweepRunner(
        s3_bucket=CONFIG.get('s3_bucket'),
        s3_prefix=CONFIG.get('s3_prefix', 'abm_sweep_results')
    )
    
    # Run parameter sweep
    logger.info("Starting parameter sweep...")
    results_df = runner.run_parameter_sweep(
        informality_rates=CONFIG['informality_rates'],
        seeds=CONFIG['seeds'],
        n_processes=CONFIG['n_processes'],
        n_firms=CONFIG['n_firms'],
        n_consumers=CONFIG['n_consumers'],
        n_banks=CONFIG['n_banks'],
        max_steps=CONFIG['max_steps'],
        inflation_target=CONFIG['inflation_target'],
        initial_policy_rate=CONFIG['initial_policy_rate'],
        current_inflation=CONFIG['current_inflation']
    )
    
    # Save detailed results
    runner.save_detailed_results(results_df)
    
    # Print summary statistics
    print("\n=== PARAMETER SWEEP SUMMARY ===")
    print(f"Total simulations: {len(results_df)}")
    print(f"Successful simulations: {len(results_df[results_df['error'].isna()])}")
    print(f"Average runtime: {results_df['run_time'].mean():.2f} seconds")
    print(f"Convergence rate: {results_df['converged'].mean()*100:.1f}%")
    
    # Group by informality rate
    summary_by_informality = results_df.groupby('parameters.informality_rate').agg({
        'final_inflation': ['mean', 'std'],
        'final_policy_rate': ['mean', 'std'],
        'credit_access_gap': ['mean', 'std'],
        'converged': 'mean',
        'run_time': 'mean'
    }).round(4)
    
    print("\n=== RESULTS BY INFORMALITY RATE ===")
    print(summary_by_informality)