#!/usr/bin/env python3
"""
Parallelized Parameter Sweep for ABM Monetary Policy Model
Runs sweeps across informality rates and random seeds
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from itertools import product
import argparse
from datetime import datetime
import pickle

# Import your model classes
from model import MonetaryPolicyModel

def run_single_simulation(params):
    """
    Run a single simulation with given parameters
    
    Args:
        params (tuple): (informality_rate, seed, sim_id, max_steps)
    
    Returns:
        dict: Simulation results
    """
    informality_rate, seed, sim_id, max_steps = params
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize model with parameters
    model = MonetaryPolicyModel(
        n_firms=50,
        n_consumers=200,
        n_commercial_banks=5,
        inflation_target=0.02,
        initial_policy_rate=0.03,
        informality_rate=informality_rate,
        current_inflation=0.12
    )
    
    # Track convergence
    converged = False
    convergence_time = None
    
    # Run simulation
    for step in range(max_steps):
        model.step()
        
        # Check for convergence (inflation stable around target)
        if step > 50 and not converged:
            if len(model.inflation_history) >= 36:
                recent_inflation = model.inflation_history[-36:]
                if all(0.01 <= val <= 0.03 for val in recent_inflation):
                    converged = True
                    convergence_time = step
                    break
    
    # Collect final metrics
    final_inflation = model.current_inflation
    final_policy_rate = model.policy_rate
    final_output_gap = model.output_gap
    
    # Get sector analysis
    sector_analysis = model.get_sector_analysis()
    
    # Calculate stability metrics
    inflation_volatility = np.std(model.inflation_history[-50:]) if len(model.inflation_history) >= 50 else np.std(model.inflation_history)
    policy_rate_volatility = np.std(model.policy_rate_history[-50:]) if len(model.policy_rate_history) >= 50 else np.std(model.policy_rate_history)
    
    # Banking sector metrics
    avg_lending_rate = np.mean([bank.lending_rate for bank in model.commercial_banks])
    total_formal_loans = sum([bank.formal_loans for bank in model.commercial_banks])
    total_informal_loans = sum([bank.informal_loans for bank in model.commercial_banks])
    credit_access_gap = model._calculate_credit_gap()
    
    # Production metrics
    formal_production = sum([f.production for f in model.firms if not f.is_informal])
    informal_production = sum([f.production for f in model.firms if f.is_informal])
    total_production = formal_production + informal_production
    
    # Consumption metrics
    formal_consumption = sum([c.consumption for c in model.consumers if not c.is_informal])
    informal_consumption = sum([c.consumption for c in model.consumers if c.is_informal])
    total_consumption = formal_consumption + informal_consumption
    
    return {
        'sim_id': sim_id,
        'informality_rate': informality_rate,
        'seed': seed,
        'steps_run': len(model.inflation_history),
        'converged': converged,
        'convergence_time': convergence_time,
        
        # Final state
        'final_inflation': final_inflation,
        'final_policy_rate': final_policy_rate,
        'final_output_gap': final_output_gap,
        'avg_lending_rate': avg_lending_rate,
        
        # Volatility measures
        'inflation_volatility': inflation_volatility,
        'policy_rate_volatility': policy_rate_volatility,
        
        # Sector metrics
        'formal_firms_count': sector_analysis['formal_firms']['count'],
        'informal_firms_count': sector_analysis['informal_firms']['count'],
        'formal_consumers_count': sector_analysis['formal_consumers']['count'],
        'informal_consumers_count': sector_analysis['informal_consumers']['count'],
        
        # Production and consumption
        'formal_production': formal_production,
        'informal_production': informal_production,
        'total_production': total_production,
        'production_ratio_formal': formal_production / total_production if total_production > 0 else 0,
        
        'formal_consumption': formal_consumption,
        'informal_consumption': informal_consumption,
        'total_consumption': total_consumption,
        'consumption_ratio_formal': formal_consumption / total_consumption if total_consumption > 0 else 0,
        
        # Credit metrics
        'total_formal_loans': total_formal_loans,
        'total_informal_loans': total_informal_loans,
        'credit_access_gap': credit_access_gap,
        'formal_credit_ratio': total_formal_loans / (total_formal_loans + total_informal_loans) if (total_formal_loans + total_informal_loans) > 0 else 0,
        
        # Sector averages
        'avg_formal_firm_production': sector_analysis['formal_firms']['avg_production'],
        'avg_informal_firm_production': sector_analysis['informal_firms']['avg_production'],
        'avg_formal_firm_price': sector_analysis['formal_firms']['avg_price'],
        'avg_informal_firm_price': sector_analysis['informal_firms']['avg_price'],
        'avg_formal_credit_access': sector_analysis['formal_firms']['avg_credit_access'],
        'avg_informal_credit_access': sector_analysis['informal_firms']['avg_credit_access'],
        
        # Time series data (last 100 steps for analysis)
        'inflation_history': model.inflation_history[-100:],
        'policy_rate_history': model.policy_rate_history[-100:],
        'output_gap_history': model.output_gap_history[-100:],
        'informality_history': model.informality_history[-100:] if len(model.informality_history) >= 100 else model.informality_history
    }

def create_parameter_combinations(informality_rates, n_seeds):
    """
    Create all parameter combinations for the sweep
    
    Args:
        informality_rates (list): List of informality rates to test
        n_seeds (int): Number of random seeds per parameter set
    
    Returns:
        list: List of parameter tuples
    """
    seeds = list(range(n_seeds))
    combinations = []
    
    sim_id = 0
    for informality_rate in informality_rates:
        for seed in seeds:
            combinations.append((informality_rate, seed, sim_id, 200))  # max_steps = 200
            sim_id += 1
    
    return combinations

def run_parameter_sweep(informality_rates, n_seeds=100, n_processes=None, output_dir='results'):
    """
    Run the full parameter sweep
    
    Args:
        informality_rates (list): List of informality rates to test
        n_seeds (int): Number of random seeds per parameter set
        n_processes (int): Number of processes to use (None for auto-detect)
        output_dir (str): Directory to save results
    """
    
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Starting parameter sweep with {len(informality_rates)} informality rates and {n_seeds} seeds each")
    print(f"Total simulations: {len(informality_rates) * n_seeds}")
    print(f"Using {n_processes} processes")
    
    # Create parameter combinations
    param_combinations = create_parameter_combinations(informality_rates, n_seeds)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulations in parallel
    start_time = datetime.now()
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, param_combinations)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"Parameter sweep completed in {duration}")
    print(f"Processed {len(results)} simulations")
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV for easy analysis
    csv_filename = os.path.join(output_dir, f'abm_sweep_results_{timestamp}.csv')
    df_results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # Save as pickle for full data (including time series)
    pickle_filename = os.path.join(output_dir, f'abm_sweep_results_full_{timestamp}.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to {pickle_filename}")
    
    # Save metadata
    metadata = {
        'informality_rates': informality_rates,
        'n_seeds': n_seeds,
        'n_processes': n_processes,
        'total_simulations': len(results),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'csv_file': csv_filename,
        'pickle_file': pickle_filename
    }
    
    metadata_filename = os.path.join(output_dir, f'sweep_metadata_{timestamp}.json')
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_filename}")
    
    return df_results, results

def main():
    parser = argparse.ArgumentParser(description='Run ABM Parameter Sweep')
    parser.add_argument('--min_informality', type=float, default=0.1, 
                       help='Minimum informality rate (default: 0.1)')
    parser.add_argument('--max_informality', type=float, default=0.8, 
                       help='Maximum informality rate (default: 0.8)')
    parser.add_argument('--informality_steps', type=int, default=8, 
                       help='Number of informality rate steps (default: 8)')
    parser.add_argument('--n_seeds', type=int, default=100, 
                       help='Number of random seeds per parameter set (default: 100)')
    parser.add_argument('--n_processes', type=int, default=None, 
                       help='Number of processes to use (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    # Create informality rate range
    informality_rates = np.linspace(args.min_informality, args.max_informality, args.informality_steps).tolist()
    
    print("Parameter Sweep Configuration:")
    print(f"  Informality rates: {informality_rates}")
    print(f"  Seeds per rate: {args.n_seeds}")
    print(f"  Total simulations: {len(informality_rates) * args.n_seeds}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run the sweep
    df_results, full_results = run_parameter_sweep(
        informality_rates=informality_rates,
        n_seeds=args.n_seeds,
        n_processes=args.n_processes,
        output_dir=args.output_dir
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Successful runs: {len(df_results)}")
    print(f"  Convergence rate: {df_results['converged'].mean():.2%}")
    print(f"  Average convergence time: {df_results[df_results['converged']]['convergence_time'].mean():.1f} steps")
    
    # Quick analysis by informality rate
    summary_by_informality = df_results.groupby('informality_rate').agg({
        'converged': 'mean',
        'convergence_time': 'mean',
        'final_inflation': ['mean', 'std'],
        'inflation_volatility': 'mean',
        'credit_access_gap': 'mean',
        'production_ratio_formal': 'mean'
    }).round(4)
    
    print("\nSummary by Informality Rate:")
    print(summary_by_informality)

if __name__ == "__main__":
    main()
