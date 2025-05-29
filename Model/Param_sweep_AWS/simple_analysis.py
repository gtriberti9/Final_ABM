#!/usr/bin/env python3
"""
Simple analysis script for ABM parameter sweep results
Handles the data structure more robustly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Optional

def find_latest_results_file(results_dir: str = "sweep_results") -> Optional[str]:
    """Find the most recent results file"""
    csv_files = glob.glob(os.path.join(results_dir, "parameter_sweep_results_*.csv"))
    if csv_files:
        return max(csv_files, key=os.path.getctime)
    return None

def load_and_clean_results(file_path: str) -> pd.DataFrame:
    """Load and clean results data"""
    print(f"Loading results from: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Print column names to debug
    print("Columns in dataset:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Extract parameters if needed
    if 'parameters.informality_rate' in df.columns:
        df['informality_rate'] = df['parameters.informality_rate']
        df['seed'] = df['parameters.seed']
    elif 'informality_rate' not in df.columns:
        print("Warning: Could not find informality_rate column")
        print("Available columns:", list(df.columns))
    
    print(f"Cleaned dataset: {len(df)} simulations")
    return df

def create_summary_analysis(df: pd.DataFrame):
    """Create summary analysis and basic plots"""
    
    if 'informality_rate' not in df.columns:
        print("Error: informality_rate column not found")
        return
    
    print("\n" + "="*50)
    print("PARAMETER SWEEP ANALYSIS SUMMARY")
    print("="*50)
    
    # Basic statistics
    print(f"Total simulations: {len(df)}")
    print(f"Informality rates tested: {sorted(df['informality_rate'].unique())}")
    print(f"Seeds per rate: {len(df) // len(df['informality_rate'].unique())}")
    
    # Group by informality rate
    summary = df.groupby('informality_rate').agg({
        'final_inflation': ['mean', 'std', 'min', 'max'],
        'final_policy_rate': ['mean', 'std'],
        'credit_access_gap': ['mean', 'std'],
        'total_production': ['mean', 'std'],
        'converged': ['mean', 'count'] if 'converged' in df.columns else ['count'],
    }).round(4)
    
    print("\nSUMMARY BY INFORMALITY RATE:")
    print(summary)
    
    # Create plots
    create_basic_plots(df)
    
    return summary

def create_basic_plots(df: pd.DataFrame):
    """Create basic visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ABM Parameter Sweep Results', fontsize=16, fontweight='bold')
    
    # Calculate means by informality rate
    grouped = df.groupby('informality_rate').agg({
        'final_inflation': 'mean',
        'final_policy_rate': 'mean',
        'credit_access_gap': 'mean',
        'total_production': 'mean'
    })
    
    informality_rates = grouped.index
    
    # 1. Final Inflation
    axes[0, 0].plot(informality_rates, grouped['final_inflation'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='Target (2%)')
    axes[0, 0].set_title('Final Inflation Rate')
    axes[0, 0].set_ylabel('Inflation Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Policy Rate
    axes[0, 1].plot(informality_rates, grouped['final_policy_rate'], 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Final Policy Rate')
    axes[0, 1].set_ylabel('Policy Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Credit Access Gap
    axes[1, 0].plot(informality_rates, grouped['credit_access_gap'], 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Credit Access Gap')
    axes[1, 0].set_ylabel('Gap (Formal - Informal)')
    axes[1, 0].set_xlabel('Informality Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Total Production
    axes[1, 1].plot(informality_rates, grouped['total_production'], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_title('Total Production')
    axes[1, 1].set_ylabel('Production')
    axes[1, 1].set_xlabel('Informality Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'abm_results_summary.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

def print_key_findings(df: pd.DataFrame):
    """Print key findings from the analysis"""
    
    if 'informality_rate' not in df.columns:
        return
    
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    # Group by informality rate
    grouped = df.groupby('informality_rate').agg({
        'final_inflation': 'mean',
        'final_policy_rate': 'mean',
        'credit_access_gap': 'mean',
        'total_production': 'mean'
    })
    
    # Find extremes
    min_inflation_rate = grouped['final_inflation'].idxmin()
    max_inflation_rate = grouped['final_inflation'].idxmax()
    
    print(f"1. INFLATION DYNAMICS:")
    print(f"   - Lowest average inflation ({grouped.loc[min_inflation_rate, 'final_inflation']:.3f}) at informality rate: {min_inflation_rate:.1f}")
    print(f"   - Highest average inflation ({grouped.loc[max_inflation_rate, 'final_inflation']:.3f}) at informality rate: {max_inflation_rate:.1f}")
    
    # Correlation analysis
    corr_inflation = df[['informality_rate', 'final_inflation']].corr().iloc[0, 1]
    corr_credit = df[['informality_rate', 'credit_access_gap']].corr().iloc[0, 1]
    corr_production = df[['informality_rate', 'total_production']].corr().iloc[0, 1]
    
    print(f"\n2. CORRELATIONS WITH INFORMALITY:")
    print(f"   - Inflation correlation: {corr_inflation:.3f}")
    print(f"   - Credit gap correlation: {corr_credit:.3f}")
    print(f"   - Production correlation: {corr_production:.3f}")
    
    print(f"\n3. CREDIT ACCESS:")
    min_gap = grouped['credit_access_gap'].min()
    max_gap = grouped['credit_access_gap'].max()
    print(f"   - Credit gap ranges from {min_gap:.3f} to {max_gap:.3f}")
    
    if 'converged' in df.columns:
        conv_rate = df['converged'].mean()
        print(f"\n4. CONVERGENCE:")
        print(f"   - Overall convergence rate: {conv_rate:.1%}")

def main():
    """Main analysis function"""
    
    # Find results file
    results_file = find_latest_results_file()
    
    if not results_file:
        print("No results file found in sweep_results/ directory")
        print("Make sure you've run the parameter sweep first")
        return
    
    try:
        # Load and analyze data
        df = load_and_clean_results(results_file)
        
        if df is None or len(df) == 0:
            print("No data to analyze")
            return
        
        # Create summary
        summary = create_summary_analysis(df)
        
        # Print key findings
        print_key_findings(df)
        
        # Save summary to CSV
        if summary is not None:
            summary_file = 'analysis_summary.csv'
            summary.to_csv(summary_file)
            print(f"\nDetailed summary saved to: {summary_file}")
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("This might be due to data format issues.")
        print("Try checking the structure of your results file manually.")

if __name__ == "__main__":
    main()