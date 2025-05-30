#!/usr/bin/env python3
"""
Simple Parameter Sweep for ABM Model
Tests how different informality rates affect economic outcomes

This script:
1. Runs multiple simulations with different informality rates
2. Uses parallel processing to speed things up
3. Saves results to CSV for easy analysis
4. Creates simple plots to visualize results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime
import os

# Import our model
from model_simple import MonetaryPolicyModel


def run_single_simulation(params):
    """
    Run one simulation with given parameters
    
    Args:
        params: tuple of (informality_rate, random_seed, simulation_id)
    
    Returns:
        dict: Results from this simulation
    """
    informality_rate, seed, sim_id = params
    
    # Set random seed for reproducible results
    np.random.seed(seed)
    
    # Create model with these parameters
    model = MonetaryPolicyModel(
        n_firms=50,                    # Number of firms
        n_consumers=200,               # Number of consumers  
        n_banks=5,                     # Number of banks
        inflation_target=0.02,         # 2% inflation target
        initial_rate=0.03,             # 3% starting interest rate
        informality_rate=informality_rate  # This is what we're testing
    )
    
    # Run simulation for up to 200 steps
    max_steps = 200
    converged = False
    convergence_step = None
    
    for step in range(max_steps):
        model.step()
        
        # Check if inflation has stabilized near target
        if step > 50:  # Wait at least 50 steps
            # Look at last 30 steps
            if len(model.inflation_history) >= 30:
                recent_inflation = model.inflation_history[-30:]
                # Check if all recent inflation is between 1% and 3%
                if all(0.01 <= val <= 0.03 for val in recent_inflation):
                    converged = True
                    convergence_step = step
                    break
    
    # Collect results from this simulation
    results = {
        'sim_id': sim_id,
        'informality_rate': informality_rate,
        'random_seed': seed,
        'steps_run': model.time_step,
        'converged': converged,
        'convergence_step': convergence_step,
        
        # Final economic state
        'final_inflation': model.current_inflation,
        'final_policy_rate': model.policy_rate,
        'final_output_gap': model.output_gap,
        
        # Economic performance metrics
        'inflation_volatility': np.std(model.inflation_history[-50:]) if len(model.inflation_history) >= 50 else np.std(model.inflation_history),
        'avg_lending_rate': np.mean([bank.lending_rate for bank in model.banks]),
        
        # Production and consumption
        'total_production': sum(firm.production for firm in model.firms),
        'total_consumption': sum(consumer.consumption for consumer in model.consumers),
        'avg_firm_price': np.mean([firm.price for firm in model.firms]),
        
        # Informal sector statistics
        'informal_firms_count': len([f for f in model.firms if f.is_informal]),
        'informal_consumers_count': len([c for c in model.consumers if c.is_informal]),
        'informal_firms_pct': len([f for f in model.firms if f.is_informal]) / len(model.firms) * 100,
        'informal_consumers_pct': len([c for c in model.consumers if c.is_informal]) / len(model.consumers) * 100,
        
        # Banking sector
        'total_formal_loans': sum(bank.formal_loans for bank in model.banks),
        'total_informal_loans': sum(bank.informal_loans for bank in model.banks),
        
        # Sector productivity comparison
        'formal_firms_avg_production': np.mean([f.production for f in model.firms if not f.is_informal]) if any(not f.is_informal for f in model.firms) else 0,
        'informal_firms_avg_production': np.mean([f.production for f in model.firms if f.is_informal]) if any(f.is_informal for f in model.firms) else 0
    }
    
    return results


def create_parameter_combinations(informality_rates, seeds_per_rate):
    """
    Create all combinations of parameters to test
    
    Args:
        informality_rates: List of informality rates to test (e.g., [0.1, 0.2, 0.3])
        seeds_per_rate: How many random seeds to use per rate (for statistical reliability)
    
    Returns:
        List of parameter tuples
    """
    combinations = []
    sim_id = 0
    
    for informality_rate in informality_rates:
        for seed in range(seeds_per_rate):
            combinations.append((informality_rate, seed, sim_id))
            sim_id += 1
    
    return combinations


def run_parameter_sweep(informality_rates, seeds_per_rate=20, n_processes=None, output_dir='results'):
    """
    Run the complete parameter sweep
    
    Args:
        informality_rates: List of informality rates to test
        seeds_per_rate: Number of random seeds per rate (more = more reliable results)
        n_processes: Number of CPU cores to use (None = use all available)
        output_dir: Where to save results
    """
    
    # Use all available CPU cores if not specified
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Starting parameter sweep...")
    print(f"Testing informality rates: {informality_rates}")
    print(f"Seeds per rate: {seeds_per_rate}")
    print(f"Total simulations: {len(informality_rates) * seeds_per_rate}")
    print(f"Using {n_processes} CPU cores")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all parameter combinations
    param_combinations = create_parameter_combinations(informality_rates, seeds_per_rate)
    
    # Run simulations in parallel
    print("\nRunning simulations...")
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        # Show progress every few seconds
        results = pool.map(run_single_simulation, param_combinations)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Simulations completed in {total_time:.1f} seconds")
    print(f"Average time per simulation: {total_time / len(results):.2f} seconds")
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f'abm_results_{timestamp}.csv')
    df.to_csv(csv_filename, index=False)
    
    print(f"Results saved to: {csv_filename}")
    
    return df, csv_filename


def analyze_results(df, output_dir):
    """
    Analyze the results and create summary statistics and plots
    
    Args:
        df: DataFrame with simulation results
        output_dir: Where to save analysis outputs
    """
    
    print("\nAnalyzing results...")
    
    # Summary statistics by informality rate
    summary = df.groupby('informality_rate').agg({
        'converged': 'mean',  # What fraction converged
        'convergence_step': 'mean',  # Average steps to converge
        'final_inflation': ['mean', 'std'],  # Average and variability of final inflation
        'inflation_volatility': 'mean',  # How volatile was inflation
        'total_production': 'mean',  # Economic output
        'total_consumption': 'mean',  # Economic demand
        'formal_firms_avg_production': 'mean',  # Formal sector productivity
        'informal_firms_avg_production': 'mean',  # Informal sector productivity
        'total_formal_loans': 'mean',  # Credit to formal sector
        'total_informal_loans': 'mean'  # Credit to informal sector
    }).round(4)
    
    print("\nSummary by Informality Rate:")
    print(summary)
    
    # Create visualizations
    create_plots(df, output_dir)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f'summary_{timestamp}.csv')
    summary.to_csv(summary_file)
    
    print(f"Summary saved to: {summary_file}")
    
    return summary


def create_plots(df, output_dir):
    """
    Create visualizations showing how informality affects economic outcomes
    """
    
    print("Creating plots...")
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Impact of Informality Rate on Economic Outcomes', fontsize=16, fontweight='bold')
    
    # Group data by informality rate for plotting
    grouped = df.groupby('informality_rate')
    
    # Plot 1: Convergence Rate
    convergence_by_rate = grouped['converged'].mean()
    axes[0, 0].plot(convergence_by_rate.index, convergence_by_rate.values, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Convergence Rate vs Informality Rate')
    axes[0, 0].set_xlabel('Informality Rate')
    axes[0, 0].set_ylabel('Fraction of Simulations that Converged')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Final Inflation
    inflation_by_rate = grouped['final_inflation'].mean()
    axes[0, 1].plot(inflation_by_rate.index, inflation_by_rate.values, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Target (2%)')
    axes[0, 1].set_title('Average Final Inflation vs Informality Rate')
    axes[0, 1].set_xlabel('Informality Rate')
    axes[0, 1].set_ylabel('Final Inflation Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Inflation Volatility
    volatility_by_rate = grouped['inflation_volatility'].mean()
    axes[0, 2].plot(volatility_by_rate.index, volatility_by_rate.values, 'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 2].set_title('Inflation Volatility vs Informality Rate')
    axes[0, 2].set_xlabel('Informality Rate')
    axes[0, 2].set_ylabel('Inflation Volatility (Std Dev)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Economic Output
    production_by_rate = grouped['total_production'].mean()
    consumption_by_rate = grouped['total_consumption'].mean()
    axes[1, 0].plot(production_by_rate.index, production_by_rate.values, 'o-', label='Production', linewidth=2, markersize=8)
    axes[1, 0].plot(consumption_by_rate.index, consumption_by_rate.values, 'o-', label='Consumption', linewidth=2, markersize=8)
    axes[1, 0].set_title('Economic Activity vs Informality Rate')
    axes[1, 0].set_xlabel('Informality Rate')
    axes[1, 0].set_ylabel('Total Economic Activity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Sector Productivity Gap
    formal_prod = grouped['formal_firms_avg_production'].mean()
    informal_prod = grouped['informal_firms_avg_production'].mean()
    productivity_gap = formal_prod - informal_prod
    
    axes[1, 1].plot(productivity_gap.index, productivity_gap.values, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_title('Formal-Informal Productivity Gap vs Informality Rate')
    axes[1, 1].set_xlabel('Informality Rate')
    axes[1, 1].set_ylabel('Productivity Gap (Formal - Informal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Credit Allocation
    formal_loans = grouped['total_formal_loans'].mean()
    informal_loans = grouped['total_informal_loans'].mean()
    
    axes[1, 2].plot(formal_loans.index, formal_loans.values, 'o-', label='Formal Sector Loans', linewidth=2, markersize=8)
    axes[1, 2].plot(informal_loans.index, informal_loans.values, 'o-', label='Informal Sector Loans', linewidth=2, markersize=8)
    axes[1, 2].set_title('Credit Allocation vs Informality Rate')
    axes[1, 2].set_xlabel('Informality Rate')
    axes[1, 2].set_ylabel('Total Loans Outstanding')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_dir, f'analysis_plots_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plot_file}")
    
    # Create a simple text summary
    create_text_summary(df, output_dir)


def create_text_summary(df, output_dir):
    """
    Create a simple text summary of key findings
    """
    
    # Compare low vs high informality
    low_informality = df[df['informality_rate'] <= 0.2]
    high_informality = df[df['informality_rate'] >= 0.6]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f'key_findings_{timestamp}.txt')
    
    with open(summary_file, 'w') as f:
        f.write("ABM Parameter Sweep - Key Findings\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total simulations run: {len(df):,}\n")
        f.write(f"Informality rates tested: {sorted(df['informality_rate'].unique())}\n")
        f.write(f"Seeds per rate: {df.groupby('informality_rate').size().iloc[0]}\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 20 + "\n\n")
        
        # Convergence comparison
        low_conv = low_informality['converged'].mean()
        high_conv = high_informality['converged'].mean()
        f.write(f"1. CONVERGENCE:\n")
        f.write(f"   Low informality (≤20%): {low_conv:.1%} of simulations converged\n")
        f.write(f"   High informality (≥60%): {high_conv:.1%} of simulations converged\n")
        f.write(f"   Impact: {((low_conv - high_conv) / high_conv * 100):+.0f}% change\n\n")
        
        # Inflation control
        low_inf = low_informality['final_inflation'].mean()
        high_inf = high_informality['final_inflation'].mean()
        f.write(f"2. INFLATION CONTROL:\n")
        f.write(f"   Low informality: {low_inf:.2%} average final inflation\n")
        f.write(f"   High informality: {high_inf:.2%} average final inflation\n")
        f.write(f"   Target inflation: 2.00%\n")
        f.write(f"   Distance from target: Low={abs(low_inf-0.02):.2%}, High={abs(high_inf-0.02):.2%}\n\n")
        
        # Economic volatility
        low_vol = low_informality['inflation_volatility'].mean()
        high_vol = high_informality['inflation_volatility'].mean()
        f.write(f"3. ECONOMIC STABILITY:\n")
        f.write(f"   Low informality: {low_vol:.4f} inflation volatility\n")
        f.write(f"   High informality: {high_vol:.4f} inflation volatility\n")
        f.write(f"   Impact: {((high_vol - low_vol) / low_vol * 100):+.0f}% increase in volatility\n\n")
        
        # Production
        low_prod = low_informality['total_production'].mean()
        high_prod = high_informality['total_production'].mean()
        f.write(f"4. ECONOMIC OUTPUT:\n")
        f.write(f"   Low informality: {low_prod:.1f} average total production\n")
        f.write(f"   High informality: {high_prod:.1f} average total production\n")
        f.write(f"   Impact: {((high_prod - low_prod) / low_prod * 100):+.1f}% change in output\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 20 + "\n")
        f.write("Higher informality rates appear to:\n")
        if high_conv < low_conv:
            f.write("- Reduce monetary policy effectiveness (lower convergence)\n")
        if abs(high_inf - 0.02) > abs(low_inf - 0.02):
            f.write("- Make inflation targeting more difficult\n")
        if high_vol > low_vol:
            f.write("- Increase economic volatility and instability\n")
        f.write("- Create challenges for traditional monetary policy transmission\n")
    
    print(f"Key findings saved to: {summary_file}")


def main():
    """
    Main function - run the parameter sweep with command line options
    """
    
    parser = argparse.ArgumentParser(description='Run ABM Parameter Sweep - Simple Version')
    parser.add_argument('--min_informality', type=float, default=0.1, 
                       help='Minimum informality rate (default: 0.1)')
    parser.add_argument('--max_informality', type=float, default=0.8, 
                       help='Maximum informality rate (default: 0.8)')
    parser.add_argument('--num_rates', type=int, default=6, 
                       help='Number of informality rates to test (default: 6)')
    parser.add_argument('--seeds_per_rate', type=int, default=20, 
                       help='Number of random seeds per rate (default: 20)')
    parser.add_argument('--n_processes', type=int, default=None, 
                       help='Number of CPU cores to use (default: all available)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    # Create list of informality rates to test
    informality_rates = np.linspace(args.min_informality, args.max_informality, args.num_rates).tolist()
    
    print("Simple ABM Parameter Sweep")
    print("=" * 40)
    print(f"Informality rates to test: {[f'{r:.1f}' for r in informality_rates]}")
    print(f"Seeds per rate: {args.seeds_per_rate}")
    print(f"Total simulations: {len(informality_rates) * args.seeds_per_rate}")
    print(f"Output directory: {args.output_dir}")
    
    # Run the parameter sweep
    results_df, csv_file = run_parameter_sweep(
        informality_rates=informality_rates,
        seeds_per_rate=args.seeds_per_rate,
        n_processes=args.n_processes,
        output_dir=args.output_dir
    )
    
    # Analyze results
    summary = analyze_results(results_df, args.output_dir)
    
    # Print quick summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Successfully completed {len(results_df)} simulations")
    print(f"Overall convergence rate: {results_df['converged'].mean():.1%}")
    print(f"Average final inflation: {results_df['final_inflation'].mean():.2%}")
    print(f"Results saved in: {args.output_dir}")
    print("\nParameter sweep completed successfully!")


if __name__ == "__main__":
    main()