#!/usr/bin/env python3
"""
Quick Analysis Script for ABM Results
For smaller datasets or when Spark is not available
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze(csv_file, output_dir='quick_analysis'):
    """
    Load CSV and perform quick analysis
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} simulation results")
    print(f"Informality rates tested: {sorted(df['informality_rate'].unique())}")
    print(f"Seeds per rate: {len(df[df['informality_rate'] == df['informality_rate'].iloc[0]])}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    print("\nComputing basic statistics...")
    
    # Overall stats
    overall_stats = {
        'total_simulations': len(df),
        'convergence_rate': df['converged'].mean(),
        'avg_inflation': df['final_inflation'].mean(),
        'std_inflation': df['final_inflation'].std(),
        'avg_credit_gap': df['credit_access_gap'].mean(),
        'avg_formal_production_ratio': df['production_ratio_formal'].mean()
    }
    
    # By informality rate
    by_informality = df.groupby('informality_rate').agg({
        'converged': ['count', 'mean'],
        'convergence_time': 'mean',
        'final_inflation': ['mean', 'std'],
        'inflation_volatility': 'mean',
        'credit_access_gap': 'mean',
        'production_ratio_formal': 'mean',
        'formal_credit_ratio': 'mean'
    }).round(4)
    
    # Flatten column names
    by_informality.columns = ['_'.join(col).strip() for col in by_informality.columns.values]
    by_informality = by_informality.reset_index()
    
    # Print key findings
    print("\nKey Findings:")
    print(f"  Overall convergence rate: {overall_stats['convergence_rate']:.1%}")
    print(f"  Average final inflation: {overall_stats['avg_inflation']:.4f} ({overall_stats['avg_inflation']*100:.2f}%)")
    print(f"  Average credit access gap: {overall_stats['avg_credit_gap']:.4f}")
    
    low_inf = by_informality[by_informality['informality_rate'] <= 0.2].iloc[0] if len(by_informality[by_informality['informality_rate'] <= 0.2]) > 0 else None
    high_inf = by_informality[by_informality['informality_rate'] >= 0.7].iloc[0] if len(by_informality[by_informality['informality_rate'] >= 0.7]) > 0 else None
    
    if low_inf is not None and high_inf is not None:
        print(f"\n  Impact of informality:")
        print(f"    Low informality ({low_inf['informality_rate']:.1f}): {low_inf['converged_mean']:.1%} convergence")
        print(f"    High informality ({high_inf['informality_rate']:.1f}): {high_inf['converged_mean']:.1%} convergence")
        print(f"    Credit gap increase: {low_inf['credit_access_gap_mean']:.4f} → {high_inf['credit_access_gap_mean']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Main analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ABM Parameter Sweep Results: Impact of Informality', fontsize=16, fontweight='bold')
    
    # 1. Convergence rate
    axes[0, 0].plot(by_informality['informality_rate'], by_informality['converged_mean'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Convergence Rate by Informality Rate')
    axes[0, 0].set_xlabel('Informality Rate')
    axes[0, 0].set_ylabel('Convergence Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Final inflation
    axes[0, 1].plot(by_informality['informality_rate'], by_informality['final_inflation_mean'], 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Target (2%)')
    axes[0, 1].set_title('Average Final Inflation')
    axes[0, 1].set_xlabel('Informality Rate')
    axes[0, 1].set_ylabel('Final Inflation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Inflation volatility
    axes[0, 2].plot(by_informality['informality_rate'], by_informality['inflation_volatility_mean'], 'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 2].set_title('Inflation Volatility')
    axes[0, 2].set_xlabel('Informality Rate')
    axes[0, 2].set_ylabel('Inflation Volatility')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Credit access gap
    axes[1, 0].plot(by_informality['informality_rate'], by_informality['credit_access_gap_mean'], 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 0].set_title('Credit Access Gap')
    axes[1, 0].set_xlabel('Informality Rate')
    axes[1, 0].set_ylabel('Credit Access Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Formal production ratio
    axes[1, 1].plot(by_informality['informality_rate'], by_informality['production_ratio_formal_mean'], 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 1].set_title('Formal Sector Production Share')
    axes[1, 1].set_xlabel('Informality Rate')
    axes[1, 1].set_ylabel('Formal Production Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Formal credit ratio
    axes[1, 2].plot(by_informality['informality_rate'], by_informality['formal_credit_ratio_mean'], 'o-', color='brown', linewidth=2, markersize=8)
    axes[1, 2].set_title('Formal Credit Share')
    axes[1, 2].set_xlabel('Informality Rate')
    axes[1, 2].set_ylabel('Formal Credit Ratio')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_dir, f'quick_analysis_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Create informality categories for box plots
    df['inf_category'] = pd.cut(df['informality_rate'], 
                               bins=[0, 0.3, 0.6, 1.0], 
                               labels=['Low (0.1-0.3)', 'Medium (0.3-0.6)', 'High (0.6-0.8)'])
    
    # Box plots
    sns.boxplot(data=df, x='inf_category', y='final_inflation', ax=axes[0, 0])
    axes[0, 0].set_title('Inflation Distribution by Informality Level')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='inf_category', y='inflation_volatility', ax=axes[0, 1])
    axes[0, 1].set_title('Inflation Volatility Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='inf_category', y='credit_access_gap', ax=axes[1, 0])
    axes[1, 0].set_title('Credit Access Gap Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='inf_category', y='production_ratio_formal', ax=axes[1, 1])
    axes[1, 1].set_title('Formal Production Ratio Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    dist_plot_file = os.path.join(output_dir, f'distributions_{timestamp}.png')
    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'overall_statistics': overall_stats,
        'by_informality_rate': by_informality.to_dict('records'),
        'timestamp': timestamp,
        'input_file': csv_file
    }
    
    results_file = os.path.join(output_dir, f'quick_analysis_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary CSV
    summary_file = os.path.join(output_dir, f'summary_by_informality_{timestamp}.csv')
    by_informality.to_csv(summary_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Main plot: {plot_file}")
    print(f"  Distributions: {dist_plot_file}")
    print(f"  JSON results: {results_file}")
    print(f"  Summary CSV: {summary_file}")
    
    return results, by_informality

def main():
    parser = argparse.ArgumentParser(description='Quick Analysis of ABM Results')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV results file')
    parser.add_argument('--output_dir', type=str, default='quick_analysis',
                       help='Output directory (default: quick_analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return
    
    results, summary = load_and_analyze(args.csv_file, args.output_dir)
    
    print(f"\nQuick analysis completed!")
    print(f"Check the '{args.output_dir}' directory for results.")

if __name__ == "__main__":
    main()
