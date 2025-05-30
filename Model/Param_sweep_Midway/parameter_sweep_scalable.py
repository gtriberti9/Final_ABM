#!/usr/bin/env python3
"""
Scalable Parameter Sweep for ABM with Multiple Storage Options
Supports CSV, Parquet, HDF5, and distributed storage formats
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
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
        params (tuple): (informality_rate, seed, sim_id, max_steps, output_format)
    
    Returns:
        dict: Simulation results
    """
    informality_rate, seed, sim_id, max_steps = params
    
    # Track timing
    start_time = time.time()
    
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
    
    end_time = time.time()
    runtime = end_time - start_time
    
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
        'runtime_seconds': runtime,
        
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
        
        # Time series data (last 100 steps for analysis) - only for detailed storage
        'inflation_history': model.inflation_history[-100:] if len(model.inflation_history) >= 100 else model.inflation_history,
        'policy_rate_history': model.policy_rate_history[-100:] if len(model.policy_rate_history) >= 100 else model.policy_rate_history,
        'output_gap_history': model.output_gap_history[-100:] if len(model.output_gap_history) >= 100 else model.output_gap_history
    }

def save_results_multiple_formats(results, output_dir, timestamp, metadata):
    """
    Save results in multiple formats for scalability comparison
    """
    df_results = pd.DataFrame(results)
    
    # Remove time series for lightweight formats
    df_light = df_results.drop(columns=['inflation_history', 'policy_rate_history', 'output_gap_history'])
    
    save_times = {}
    file_sizes = {}
    
    print(f"Saving {len(df_results)} results in multiple formats...")
    
    # 1. CSV (traditional, human-readable)
    start_time = time.time()
    csv_file = os.path.join(output_dir, f'abm_sweep_results_{timestamp}.csv')
    df_light.to_csv(csv_file, index=False)
    save_times['CSV'] = time.time() - start_time
    file_sizes['CSV'] = os.path.getsize(csv_file) / (1024 * 1024)  # MB
    
    # 2. Parquet (fast, compressed, columnar)
    try:
        start_time = time.time()
        parquet_file = os.path.join(output_dir, f'abm_sweep_results_{timestamp}.parquet')
        df_results.to_parquet(parquet_file, index=False, compression='snappy')
        save_times['Parquet'] = time.time() - start_time
        file_sizes['Parquet'] = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"Parquet saved: {parquet_file}")
    except Exception as e:
        print(f"Parquet save failed: {e}")
    
    # 3. HDF5 (hierarchical, good for time series)
    try:
        start_time = time.time()
        hdf5_file = os.path.join(output_dir, f'abm_sweep_results_{timestamp}.h5')
        
        # Save main results
        df_light.to_hdf(hdf5_file, key='results', mode='w', format='table', complevel=9)
        
        # Save time series separately for efficiency
        with pd.HDFStore(hdf5_file, mode='a') as store:
            for i, result in enumerate(results):
                if 'inflation_history' in result and result['inflation_history']:
                    ts_data = pd.DataFrame({
                        'step': range(len(result['inflation_history'])),
                        'inflation': result['inflation_history'],
                        'policy_rate': result['policy_rate_history'][:len(result['inflation_history'])],
                        'output_gap': result['output_gap_history'][:len(result['inflation_history'])]
                    })
                    store.put(f'timeseries/sim_{i}', ts_data, format='table')
        
        save_times['HDF5'] = time.time() - start_time
        file_sizes['HDF5'] = os.path.getsize(hdf5_file) / (1024 * 1024)
        print(f"HDF5 saved: {hdf5_file}")
    except Exception as e:
        print(f"HDF5 save failed: {e}")
    
    # 4. Feather (fast read/write)
    try:
        start_time = time.time()
        feather_file = os.path.join(output_dir, f'abm_sweep_results_{timestamp}.feather')
        df_light.to_feather(feather_file)
        save_times['Feather'] = time.time() - start_time
        file_sizes['Feather'] = os.path.getsize(feather_file) / (1024 * 1024)
        print(f"Feather saved: {feather_file}")
    except Exception as e:
        print(f"Feather save failed: {e}")
    
    # 5. Pickle (full Python objects)
    start_time = time.time()
    pickle_file = os.path.join(output_dir, f'abm_sweep_results_full_{timestamp}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_times['Pickle'] = time.time() - start_time
    file_sizes['Pickle'] = os.path.getsize(pickle_file) / (1024 * 1024)
    
    # Save performance comparison
    perf_data = []
    for fmt in save_times.keys():
        perf_data.append({
            'format': fmt,
            'save_time_seconds': save_times[fmt],
            'file_size_mb': file_sizes[fmt],
            'write_speed_mb_s': file_sizes[fmt] / save_times[fmt] if save_times[fmt] > 0 else 0
        })
    
    perf_df = pd.DataFrame(perf_data)
    perf_file = os.path.join(output_dir, f'storage_performance_{timestamp}.csv')
    perf_df.to_csv(perf_file, index=False)
    
    # Update metadata with storage info
    metadata.update({
        'storage_formats': list(save_times.keys()),
        'storage_performance': perf_data,
        'primary_csv_file': csv_file,
        'parquet_file': parquet_file if 'Parquet' in save_times else None,
        'hdf5_file': hdf5_file if 'HDF5' in save_times else None
    })
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'sweep_metadata_{timestamp}.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nStorage Performance Summary:")
    print(perf_df.to_string(index=False))
    
    return {
        'csv_file': csv_file,
        'performance_file': perf_file,
        'metadata_file': metadata_file,
        'storage_performance': perf_data
    }

def create_parameter_combinations(informality_rates, n_seeds):
    """
    Create all parameter combinations for the sweep
    """
    seeds = list(range(n_seeds))
    combinations = []
    
    sim_id = 0
    for informality_rate in informality_rates:
        for seed in seeds:
            combinations.append((informality_rate, seed, sim_id, 200))  # max_steps = 200
            sim_id += 1
    
    return combinations

def run_parameter_sweep(informality_rates, n_seeds=100, n_processes=None, output_dir='results', output_format='all'):
    """
    Run the parameter sweep with scalable storage options
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Starting scalable parameter sweep")
    print(f"Informality rates: {informality_rates}")
    print(f"Seeds per rate: {n_seeds}")
    print(f"Total simulations: {len(informality_rates) * n_seeds}")
    print(f"Using {n_processes} processes")
    print(f"Output formats: {output_format}")
    
    # Create parameter combinations
    param_combinations = create_parameter_combinations(informality_rates, n_seeds)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulations in parallel with timing
    start_time = datetime.now()
    compute_start = time.time()
    
    print(f"\nStarting simulations at {start_time.strftime('%H:%M:%S')}...")
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, param_combinations)
    
    compute_end = time.time()
    compute_time = compute_end - compute_start
    
    print(f"Simulations completed in {compute_time:.2f} seconds")
    print(f"Average time per simulation: {compute_time / len(results):.3f} seconds")
    print(f"Throughput: {len(results) / compute_time:.2f} simulations/second")
    
    # Save results in multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        'informality_rates': informality_rates,
        'n_seeds': n_seeds,
        'n_processes': n_processes,
        'total_simulations': len(results),
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'compute_time_seconds': compute_time,
        'avg_time_per_sim': compute_time / len(results),
        'throughput_sims_per_sec': len(results) / compute_time,
        'output_format': output_format
    }
    
    storage_start = time.time()
    file_info = save_results_multiple_formats(results, output_dir, timestamp, metadata)
    storage_time = time.time() - storage_start
    
    print(f"\nStorage completed in {storage_time:.2f} seconds")
    
    # Final summary
    total_time = time.time() - compute_start
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    print(f"  Computation: {compute_time:.2f}s ({compute_time/total_time:.1%})")
    print(f"  Storage: {storage_time:.2f}s ({storage_time/total_time:.1%})")
    
    return results, file_info, metadata

def benchmark_read_performance(output_dir, timestamp):
    """
    Benchmark reading performance of different formats
    """
    print("\nBenchmarking read performance...")
    
    formats = {
        'CSV': f'{output_dir}/abm_sweep_results_{timestamp}.csv',
        'Parquet': f'{output_dir}/abm_sweep_results_{timestamp}.parquet',
        'HDF5': f'{output_dir}/abm_sweep_results_{timestamp}.h5',
        'Feather': f'{output_dir}/abm_sweep_results_{timestamp}.feather',
        'Pickle': f'{output_dir}/abm_sweep_results_full_{timestamp}.pkl'
    }
    
    read_performance = []
    
    for format_name, filepath in formats.items():
        if os.path.exists(filepath):
            try:
                start_time = time.time()
                
                if format_name == 'CSV':
                    data = pd.read_csv(filepath)
                elif format_name == 'Parquet':
                    data = pd.read_parquet(filepath)
                elif format_name == 'HDF5':
                    data = pd.read_hdf(filepath, key='results')
                elif format_name == 'Feather':
                    data = pd.read_feather(filepath)
                elif format_name == 'Pickle':
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                
                read_time = time.time() - start_time
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                
                read_performance.append({
                    'format': format_name,
                    'read_time_seconds': read_time,
                    'file_size_mb': file_size,
                    'read_speed_mb_s': file_size / read_time if read_time > 0 else 0,
                    'rows_loaded': len(data) if hasattr(data, '__len__') else 'N/A'
                })
                
                print(f"  {format_name}: {read_time:.3f}s ({file_size / read_time:.1f} MB/s)")
                
            except Exception as e:
                print(f"  {format_name}: Failed to read - {e}")
    
    # Save read performance
    read_perf_df = pd.DataFrame(read_performance)
    read_perf_file = os.path.join(output_dir, f'read_performance_{timestamp}.csv')
    read_perf_df.to_csv(read_perf_file, index=False)
    
    return read_performance

def main():
    parser = argparse.ArgumentParser(description='Run Scalable ABM Parameter Sweep')
    parser.add_argument('--min_informality', type=float, default=0.1)
    parser.add_argument('--max_informality', type=float, default=0.8)
    parser.add_argument('--informality_steps', type=int, default=8)
    parser.add_argument('--n_seeds', type=int, default=100)
    parser.add_argument('--n_processes', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--output_format', type=str, default='all', 
                       choices=['csv', 'parquet', 'hdf5', 'feather', 'pickle', 'all'])
    parser.add_argument('--benchmark_read', action='store_true', 
                       help='Benchmark read performance after saving')
    
    args = parser.parse_args()
    
    # Create informality rate range
    informality_rates = np.linspace(args.min_informality, args.max_informality, args.informality_steps).tolist()
    
    print("Scalable Parameter Sweep Configuration:")
    print(f"  Informality rates: {informality_rates}")
    print(f"  Seeds per rate: {args.n_seeds}")
    print(f"  Total simulations: {len(informality_rates) * args.n_seeds}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output format: {args.output_format}")
    
    # Run the sweep
    results, file_info, metadata = run_parameter_sweep(
        informality_rates=informality_rates,
        n_seeds=args.n_seeds,
        n_processes=args.n_processes,
        output_dir=args.output_dir,
        output_format=args.output_format
    )
    
    # Benchmark read performance if requested
    if args.benchmark_read:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        read_performance = benchmark_read_performance(args.output_dir, 
                                                     file_info['metadata_file'].split('_')[-1].replace('.json', ''))
    
    # Print summary statistics
    df_results = pd.DataFrame(results)
    print("\nSummary Statistics:")
    print(f"  Successful runs: {len(df_results)}")
    print(f"  Convergence rate: {df_results['converged'].mean():.2%}")
    if df_results['converged'].any():
        print(f"  Average convergence time: {df_results[df_results['converged']]['convergence_time'].mean():.1f} steps")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    print(f"  Average runtime per simulation: {df_results['runtime_seconds'].mean():.3f} seconds")
    print(f"  Total computation time: {df_results['runtime_seconds'].sum():.2f} seconds")
    print(f"  Parallel efficiency: {df_results['runtime_seconds'].sum() / metadata['compute_time_seconds']:.2f}")
    
    print(f"\nFiles saved to: {args.output_dir}")
    print("Scalable parameter sweep completed!")

if __name__ == "__main__":
    main()