#!/usr/bin/env python3
"""
Scalability Benchmark for ABM Parameter Sweep
Tests performance across different core counts and data sizes
"""

import numpy as np
import pandas as pd
import time
import json
import os
import pickle
import argparse
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Import your model
from model import MonetaryPolicyModel

def run_single_sim_timed(params):
    """Run a single simulation with timing"""
    informality_rate, seed, sim_id, max_steps = params
    
    start_time = time.time()
    
    # Set random seed
    np.random.seed(seed)
    
    # Initialize and run model
    model = MonetaryPolicyModel(
        n_firms=50,
        n_consumers=200,
        n_commercial_banks=5,
        inflation_target=0.02,
        initial_policy_rate=0.03,
        informality_rate=informality_rate,
        current_inflation=0.12
    )
    
    # Run simulation
    for step in range(max_steps):
        model.step()
        if step > 50 and model.is_converged():
            break
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return {
        'sim_id': sim_id,
        'informality_rate': informality_rate,
        'seed': seed,
        'steps_run': len(model.inflation_history),
        'converged': model.is_converged(),
        'runtime_seconds': runtime,
        'final_inflation': model.current_inflation,
        'final_policy_rate': model.policy_rate,
        'credit_access_gap': model._calculate_credit_gap()
    }

def benchmark_cores(n_sims_list, core_counts, output_dir='benchmarks'):
    """
    Benchmark performance across different core counts and simulation sizes
    """
    print("Starting scalability benchmark...")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for n_sims in n_sims_list:
        print(f"\nBenchmarking with {n_sims} simulations...")
        
        # Create parameter combinations
        params = []
        for i in range(n_sims):
            informality_rate = np.random.uniform(0.1, 0.8)
            seed = np.random.randint(0, 10000)
            params.append((informality_rate, seed, i, 100))  # 100 max steps
        
        for n_cores in core_counts:
            if n_cores > cpu_count():
                print(f"Skipping {n_cores} cores (only {cpu_count()} available)")
                continue
                
            print(f"  Testing {n_cores} cores...")
            
            start_time = time.time()
            
            if n_cores == 1:
                # Serial execution
                sim_results = [run_single_sim_timed(p) for p in params]
            else:
                # Parallel execution
                with Pool(processes=n_cores) as pool:
                    sim_results = pool.map(run_single_sim_timed, params)
            
            end_time = time.time()
            total_runtime = end_time - start_time
            
            # Calculate metrics
            avg_sim_time = np.mean([r['runtime_seconds'] for r in sim_results])
            total_sim_time = sum([r['runtime_seconds'] for r in sim_results])
            efficiency = total_sim_time / (total_runtime * n_cores) if n_cores > 1 else 1.0
            speedup = results[0]['total_runtime'] / total_runtime if results and n_sims == results[0]['n_sims'] else 1.0
            
            benchmark_result = {
                'n_sims': n_sims,
                'n_cores': n_cores,
                'total_runtime': total_runtime,
                'avg_sim_time': avg_sim_time,
                'total_sim_time': total_sim_time,
                'efficiency': efficiency,
                'speedup': speedup,
                'sims_per_second': n_sims / total_runtime,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(benchmark_result)
            
            print(f"    Runtime: {total_runtime:.2f}s, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2%}")
    
    return results

def benchmark_storage_formats(data_sizes, output_dir='benchmarks'):
    """
    Benchmark different storage formats for scalability
    """
    print("\nBenchmarking storage formats...")
    
    storage_results = []
    
    for n_rows in data_sizes:
        print(f"Testing with {n_rows:,} rows...")
        
        # Generate sample data
        data = {
            'sim_id': range(n_rows),
            'informality_rate': np.random.uniform(0.1, 0.8, n_rows),
            'seed': np.random.randint(0, 10000, n_rows),
            'converged': np.random.choice([True, False], n_rows),
            'final_inflation': np.random.normal(0.03, 0.01, n_rows),
            'credit_access_gap': np.random.normal(0.2, 0.05, n_rows),
            'inflation_volatility': np.random.exponential(0.01, n_rows),
            'production_ratio_formal': np.random.beta(2, 2, n_rows)
        }
        df = pd.DataFrame(data)
        
        # Test different formats
        formats = {
            'CSV': ('csv', lambda: df.to_csv(f'{output_dir}/test_{n_rows}.csv', index=False)),
            'Pickle': ('pkl', lambda: df.to_pickle(f'{output_dir}/test_{n_rows}.pkl')),
            'Parquet': ('parquet', lambda: df.to_parquet(f'{output_dir}/test_{n_rows}.parquet')),
            'HDF5': ('h5', lambda: df.to_hdf(f'{output_dir}/test_{n_rows}.h5', key='data', mode='w')),
            'Feather': ('feather', lambda: df.to_feather(f'{output_dir}/test_{n_rows}.feather'))
        }
        
        for format_name, (ext, write_func) in formats.items():
            try:
                # Time writing
                start_time = time.time()
                write_func()
                write_time = time.time() - start_time
                
                # Get file size
                filename = f'{output_dir}/test_{n_rows}.{ext}'
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                
                # Time reading
                start_time = time.time()
                if format_name == 'CSV':
                    pd.read_csv(filename)
                elif format_name == 'Pickle':
                    pd.read_pickle(filename)
                elif format_name == 'Parquet':
                    pd.read_parquet(filename)
                elif format_name == 'HDF5':
                    pd.read_hdf(filename, key='data')
                elif format_name == 'Feather':
                    pd.read_feather(filename)
                read_time = time.time() - start_time
                
                storage_results.append({
                    'n_rows': n_rows,
                    'format': format_name,
                    'write_time': write_time,
                    'read_time': read_time,
                    'file_size_mb': file_size,
                    'write_speed_mb_s': file_size / write_time if write_time > 0 else 0,
                    'read_speed_mb_s': file_size / read_time if read_time > 0 else 0
                })
                
                print(f"  {format_name}: {file_size:.1f}MB, Write: {write_time:.2f}s, Read: {read_time:.2f}s")
                
                # Clean up
                os.remove(filename)
                
            except Exception as e:
                print(f"  {format_name}: Error - {e}")
    
    return storage_results

def create_scalability_plots(benchmark_results, storage_results, output_dir='benchmarks'):
    """
    Create visualizations showing scalability
    """
    print("\nCreating scalability visualizations...")
    
    # Convert to DataFrames
    bench_df = pd.DataFrame(benchmark_results)
    storage_df = pd.DataFrame(storage_results)
    
    # Create comprehensive scalability plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ABM Scalability Analysis', fontsize=16, fontweight='bold')
    
    # 1. Speedup vs Cores
    if len(bench_df) > 0:
        # Get largest dataset for speedup analysis
        max_sims = bench_df['n_sims'].max()
        speedup_data = bench_df[bench_df['n_sims'] == max_sims]
        
        axes[0, 0].plot(speedup_data['n_cores'], speedup_data['speedup'], 'o-', linewidth=2, markersize=8, label='Actual')
        axes[0, 0].plot(speedup_data['n_cores'], speedup_data['n_cores'], '--', alpha=0.7, label='Ideal Linear')
        axes[0, 0].set_title(f'Speedup vs Core Count ({max_sims:,} simulations)')
        axes[0, 0].set_xlabel('Number of Cores')
        axes[0, 0].set_ylabel('Speedup Factor')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Efficiency vs Cores
        axes[0, 1].plot(speedup_data['n_cores'], speedup_data['efficiency'], 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
        axes[0, 1].set_title('Parallel Efficiency')
        axes[0, 1].set_xlabel('Number of Cores')
        axes[0, 1].set_ylabel('Efficiency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Throughput vs Cores
        axes[0, 2].plot(speedup_data['n_cores'], speedup_data['sims_per_second'], 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 2].set_title('Simulation Throughput')
        axes[0, 2].set_xlabel('Number of Cores')
        axes[0, 2].set_ylabel('Simulations per Second')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Storage Format Comparison - Write Speed
    if len(storage_df) > 0:
        pivot_write = storage_df.pivot(index='n_rows', columns='format', values='write_time')
        for format_name in pivot_write.columns:
            axes[1, 0].plot(pivot_write.index, pivot_write[format_name], 'o-', label=format_name, linewidth=2)
        axes[1, 0].set_title('Write Performance by Format')
        axes[1, 0].set_xlabel('Number of Rows')
        axes[1, 0].set_ylabel('Write Time (seconds)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Storage Format Comparison - File Size
        pivot_size = storage_df.pivot(index='n_rows', columns='format', values='file_size_mb')
        for format_name in pivot_size.columns:
            axes[1, 1].plot(pivot_size.index, pivot_size[format_name], 'o-', label=format_name, linewidth=2)
        axes[1, 1].set_title('Storage Efficiency by Format')
        axes[1, 1].set_xlabel('Number of Rows')
        axes[1, 1].set_ylabel('File Size (MB)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Read Performance
        pivot_read = storage_df.pivot(index='n_rows', columns='format', values='read_time')
        for format_name in pivot_read.columns:
            axes[1, 2].plot(pivot_read.index, pivot_read[format_name], 'o-', label=format_name, linewidth=2)
        axes[1, 2].set_title('Read Performance by Format')
        axes[1, 2].set_xlabel('Number of Rows')
        axes[1, 2].set_ylabel('Read Time (seconds)')
        axes[1, 2].set_xscale('log')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_dir, f'scalability_analysis_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_data = []
    
    if len(bench_df) > 0:
        # Best performance metrics
        best_throughput = bench_df.loc[bench_df['sims_per_second'].idxmax()]
        best_efficiency = bench_df.loc[bench_df['efficiency'].idxmax()]
        
        summary_data.extend([
            ['Compute Performance', '', ''],
            ['Best Throughput', f"{best_throughput['sims_per_second']:.1f} sims/sec", f"{best_throughput['n_cores']} cores"],
            ['Best Efficiency', f"{best_efficiency['efficiency']:.1%}", f"{best_efficiency['n_cores']} cores"],
            ['Max Speedup Tested', f"{bench_df['speedup'].max():.2f}x", f"{bench_df['n_cores'].max()} cores"]
        ])
    
    if len(storage_df) > 0:
        # Best storage formats
        largest_dataset = storage_df[storage_df['n_rows'] == storage_df['n_rows'].max()]
        fastest_write = largest_dataset.loc[largest_dataset['write_time'].idxmin()]
        fastest_read = largest_dataset.loc[largest_dataset['read_time'].idxmin()]
        smallest_file = largest_dataset.loc[largest_dataset['file_size_mb'].idxmin()]
        
        summary_data.extend([
            ['', '', ''],
            ['Storage Performance', '', ''],
            ['Fastest Write', fastest_write['format'], f"{fastest_write['write_time']:.2f}s"],
            ['Fastest Read', fastest_read['format'], f"{fastest_read['read_time']:.2f}s"],
            ['Most Compact', smallest_file['format'], f"{smallest_file['file_size_mb']:.1f}MB"]
        ])
    
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value', 'Details'])
    summary_file = os.path.join(output_dir, f'scalability_summary_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Scalability analysis saved:")
    print(f"  Plot: {plot_file}")
    print(f"  Summary: {summary_file}")
    
    return plot_file, summary_file

def main():
    parser = argparse.ArgumentParser(description='ABM Scalability Benchmark')
    parser.add_argument('--max_sims', type=int, default=200, help='Maximum simulations to test')
    parser.add_argument('--max_cores', type=int, default=None, help='Maximum cores to test')
    parser.add_argument('--output_dir', type=str, default='benchmarks', help='Output directory')
    parser.add_argument('--storage_only', action='store_true', help='Only test storage formats')
    parser.add_argument('--compute_only', action='store_true', help='Only test compute scalability')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    max_cores = args.max_cores or min(cpu_count(), 20)
    
    benchmark_results = []
    storage_results = []
    
    if not args.storage_only:
        # Test compute scalability
        n_sims_list = [50, 100, 200] if args.max_sims >= 200 else [20, 50, 100]
        core_counts = [1, 2, 4, 8, max_cores] if max_cores >= 8 else [1, 2, max_cores]
        benchmark_results = benchmark_cores(n_sims_list, core_counts, args.output_dir)
    
    if not args.compute_only:
        # Test storage scalability
        data_sizes = [1000, 10000, 100000, 1000000]  # 1K to 1M rows
        storage_results = benchmark_storage_formats(data_sizes, args.output_dir)
    
    # Create visualizations
    plot_file, summary_file = create_scalability_plots(benchmark_results, storage_results, args.output_dir)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if benchmark_results:
        bench_file = os.path.join(args.output_dir, f'compute_benchmark_{timestamp}.json')
        with open(bench_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"Compute benchmark results: {bench_file}")
    
    if storage_results:
        storage_file = os.path.join(args.output_dir, f'storage_benchmark_{timestamp}.json')
        with open(storage_file, 'w') as f:
            json.dump(storage_results, f, indent=2)
        print(f"Storage benchmark results: {storage_file}")
    
    print(f"\nScalability analysis complete!")

if __name__ == "__main__":
    main()