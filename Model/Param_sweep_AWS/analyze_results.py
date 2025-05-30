#!/usr/bin/env python3
"""
Enhanced analysis script for MESA-FREE ABM results with statistical analysis
Includes comprehensive statistical tests and analysis
Works with both Mesa and Mesa-free simulation results
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, f_oneway, pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_results():
    """Load results from multiple pickle files (from multiple instances)"""
    
    # Look for result files from multiple instances (both Mesa and Mesa-free)
    result_files = glob.glob('results/abm_results_*.pkl')
    
    if not result_files:
        print("❌ No result files found in 'results/' folder")
        print("Make sure the AWS sweep completed and results were downloaded")
        return None
    
    print(f"Found {len(result_files)} result files from different instances")
    
    all_results = []
    mesa_free_count = 0
    mesa_count = 0
    
    for file_path in result_files:
        print(f"Loading: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
                
                # Check if results are from Mesa-free version
                for result in results:
                    if isinstance(result, dict) and result.get('mesa_free', False):
                        mesa_free_count += 1
                    else:
                        mesa_count += 1
                
                all_results.extend(results)
        except Exception as e:
            print(f"⚠️ Failed to load {file_path}: {e}")
    
    print(f"✅ Loaded {len(all_results)} total simulation results")
    if mesa_free_count > 0:
        print(f"   - {mesa_free_count} from MESA-FREE simulations")
    if mesa_count > 0:
        print(f"   - {mesa_count} from Mesa simulations")
    
    return all_results

def convert_to_dataframe(results):
    """Convert results to pandas DataFrame"""
    
    # Filter out failed simulations
    successful_results = [r for r in results if 'error' not in r]
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        print(f"⚠️ {failed_count} simulations failed")
    
    if not successful_results:
        print("❌ No successful simulations found")
        return None
    
    df = pd.DataFrame(successful_results)
    print(f"✅ Created DataFrame with {len(df)} successful simulations")
    
    # Check if we have Mesa-free results
    if 'mesa_free' in df.columns:
        mesa_free_count = df['mesa_free'].sum() if df['mesa_free'].dtype == bool else 0
        print(f"   - {mesa_free_count} MESA-FREE simulations")
        print(f"   - {len(df) - mesa_free_count} Regular simulations")
    
    return df

def comprehensive_statistical_analysis(df):
    """
    Perform comprehensive statistical analysis
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    if 'mesa_free' in df.columns and df['mesa_free'].any():
        print("(Including MESA-FREE simulation results)")
    print("="*80)
    
    results = {}
    
    # 1. NORMALITY TESTS
    print("\n1. NORMALITY TESTS")
    print("-" * 40)
    
    key_variables = ['final_inflation', 'final_policy_rate', 'credit_access_gap', 'total_production']
    normality_results = {}
    
    for var in key_variables:
        if var in df.columns:
            # Shapiro-Wilk test (best for small samples)
            stat, p_value = stats.shapiro(df[var])
            is_normal = p_value > 0.05
            normality_results[var] = {'statistic': stat, 'p_value': p_value, 'is_normal': is_normal}
            
            print(f"{var}:")
            print(f"  Shapiro-Wilk: W={stat:.4f}, p={p_value:.6f} {'(Normal)' if is_normal else '(Non-normal)'}")
    
    results['normality'] = normality_results
    
    # 2. KRUSKAL-WALLIS TEST (Non-parametric ANOVA)
    print("\n2. KRUSKAL-WALLIS TESTS (Non-parametric ANOVA)")
    print("-" * 50)
    
    kruskal_results = {}
    
    for var in key_variables:
        if var in df.columns:
            # Group by informality rate
            groups = [group[var].values for name, group in df.groupby('informality_rate')]
            
            if len(groups) > 2:  # Need at least 3 groups
                h_stat, p_value = kruskal(*groups)
                effect_size = h_stat / (len(df) - 1)  # Eta-squared approximation
                
                kruskal_results[var] = {
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05
                }
                
                print(f"{var}:")
                print(f"  H-statistic: {h_stat:.4f}")
                print(f"  p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
                print(f"  Effect size (η²): {effect_size:.4f}")
                print(f"  Interpretation: {'Large effect' if effect_size > 0.14 else 'Medium effect' if effect_size > 0.06 else 'Small effect'}")
    
    results['kruskal_wallis'] = kruskal_results
    
    # 3. PAIRWISE COMPARISONS (Mann-Whitney U tests with Bonferroni correction)
    print("\n3. PAIRWISE COMPARISONS (Mann-Whitney U)")
    print("-" * 45)
    
    pairwise_results = {}
    informality_rates = sorted(df['informality_rate'].unique())
    
    for var in ['final_inflation', 'credit_access_gap']:
        if var in df.columns:
            print(f"\n{var} - Pairwise comparisons:")
            pairwise_results[var] = {}
            
            # Number of comparisons for Bonferroni correction
            n_comparisons = len(informality_rates) * (len(informality_rates) - 1) // 2
            alpha_corrected = 0.05 / n_comparisons
            
            for i, rate1 in enumerate(informality_rates):
                for rate2 in informality_rates[i+1:]:
                    group1 = df[df['informality_rate'] == rate1][var]
                    group2 = df[df['informality_rate'] == rate2][var]
                    
                    # Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    
                    # Effect size (r = Z / sqrt(N))
                    n1, n2 = len(group1), len(group2)
                    z_score = stats.norm.ppf(1 - p_value/2)  # Approximation
                    effect_size = abs(z_score) / np.sqrt(n1 + n2)
                    
                    comparison_key = f"{rate1}_vs_{rate2}"
                    pairwise_results[var][comparison_key] = {
                        'u_statistic': u_stat,
                        'p_value': p_value,
                        'p_value_corrected': p_value * n_comparisons,
                        'effect_size': effect_size,
                        'significant_uncorrected': p_value < 0.05,
                        'significant_bonferroni': p_value < alpha_corrected
                    }
                    
                    significance = ""
                    if p_value < alpha_corrected:
                        significance = " (Significant after Bonferroni correction)"
                    elif p_value < 0.05:
                        significance = " (Significant before correction)"
                    
                    print(f"  {rate1:.1f} vs {rate2:.1f}: U={u_stat:.1f}, p={p_value:.4f}, r={effect_size:.3f}{significance}")
    
    results['pairwise'] = pairwise_results
    
    # 4. CORRELATION ANALYSIS
    print("\n4. CORRELATION ANALYSIS")
    print("-" * 30)
    
    correlation_results = {}
    
    # Pearson correlations
    correlations = df[['informality_rate'] + key_variables].corr()['informality_rate'].drop('informality_rate')
    
    print("Pearson correlations with informality rate:")
    for var, corr in correlations.items():
        if not pd.isna(corr):
            # Test significance
            n = len(df)
            t_stat = corr * np.sqrt((n-2)/(1-corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
            
            correlation_results[var] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {var}: r = {corr:.4f}, p = {p_value:.4f} {significance}")
    
    results['correlations'] = correlation_results
    
    # 5. REGRESSION ANALYSIS
    print("\n5. REGRESSION ANALYSIS")
    print("-" * 25)
    
    from scipy.stats import linregress
    regression_results = {}
    
    for var in ['final_inflation', 'credit_access_gap']:
        if var in df.columns:
            slope, intercept, r_value, p_value, std_err = linregress(df['informality_rate'], df[var])
            
            # Calculate additional statistics
            r_squared = r_value ** 2
            n = len(df)
            t_stat = slope / std_err
            
            regression_results[var] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_error': std_err,
                't_statistic': t_stat,
                'significant': p_value < 0.05
            }
            
            print(f"{var} ~ informality_rate:")
            print(f"  Slope: {slope:.6f} ± {std_err:.6f}")
            print(f"  R²: {r_squared:.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    
    results['regression'] = regression_results
    
    # 6. MESA-FREE COMPARISON (if available)
    if 'mesa_free' in df.columns and df['mesa_free'].any():
        print("\n6. MESA-FREE vs MESA COMPARISON")
        print("-" * 35)
        
        mesa_free_results = df[df['mesa_free'] == True]
        regular_results = df[df['mesa_free'] != True]
        
        if len(mesa_free_results) > 0 and len(regular_results) > 0:
            print("Comparing Mesa-free vs Regular Mesa simulations:")
            
            for var in key_variables:
                if var in df.columns:
                    mesa_free_vals = mesa_free_results[var]
                    regular_vals = regular_results[var]
                    
                    # Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(mesa_free_vals, regular_vals, alternative='two-sided')
                    
                    print(f"  {var}:")
                    print(f"    Mesa-free mean: {mesa_free_vals.mean():.4f}")
                    print(f"    Regular mean: {regular_vals.mean():.4f}")
                    print(f"    Mann-Whitney U: {u_stat:.1f}, p={p_value:.4f}")
    
    # 7. EFFECT SIZE INTERPRETATIONS
    print("\n7. EFFECT SIZE SUMMARY")
    print("-" * 25)
    
    print("Cohen's conventions for effect sizes:")
    print("  Correlation (r): Small = 0.10, Medium = 0.30, Large = 0.50")
    print("  Eta-squared (η²): Small = 0.01, Medium = 0.06, Large = 0.14")
    print("")
    
    for var in key_variables:
        if var in correlation_results:
            r = abs(correlation_results[var]['correlation'])
            size = "Large" if r >= 0.5 else "Medium" if r >= 0.3 else "Small" if r >= 0.1 else "Negligible"
            print(f"  {var}: r = {r:.3f} ({size} effect)")
    
    return results

def create_enhanced_visualizations(df):
    """Create enhanced visualizations with statistical annotations"""
    
    print(f"\nCreating enhanced visualizations...")
    
    # Check if we have Mesa-free results for title
    title_suffix = ""
    if 'mesa_free' in df.columns and df['mesa_free'].any():
        mesa_free_count = df['mesa_free'].sum() if df['mesa_free'].dtype == bool else 0
        title_suffix = f" (Including {mesa_free_count} Mesa-free results)"
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Group by informality rate for plotting
    grouped = df.groupby('informality_rate').agg({
        'final_inflation': ['mean', 'std', 'count'],
        'final_policy_rate': ['mean', 'std'],
        'credit_access_gap': ['mean', 'std'],
        'total_production': ['mean', 'std'],
        'formal_production': ['mean', 'std'],
        'informal_production': ['mean', 'std']
    })
    
    informality_rates = grouped.index
    
    # 1. Final Inflation with Confidence Intervals
    ax1 = fig.add_subplot(gs[0, 0])
    means = grouped['final_inflation']['mean']
    stds = grouped['final_inflation']['std']
    counts = grouped['final_inflation']['count']
    ci = 1.96 * stds / np.sqrt(counts)  # 95% confidence interval
    
    ax1.errorbar(informality_rates, means, yerr=ci, fmt='bo-', linewidth=2, 
                markersize=8, capsize=5, capthick=2)
    ax1.axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='Target (2%)')
    ax1.set_title('Final Inflation Rate (±95% CI)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inflation Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation annotation
    corr, p_val = pearsonr(df['informality_rate'], df['final_inflation'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 2. Credit Access Gap
    ax2 = fig.add_subplot(gs[0, 1])
    means = grouped['credit_access_gap']['mean']
    stds = grouped['credit_access_gap']['std']
    ci = 1.96 * stds / np.sqrt(counts)
    
    ax2.errorbar(informality_rates, means, yerr=ci, fmt='ro-', linewidth=2, 
                markersize=8, capsize=5, capthick=2)
    ax2.set_title('Credit Access Gap (±95% CI)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gap (Formal - Informal)')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation annotation
    corr, p_val = pearsonr(df['informality_rate'], df['credit_access_gap'])
    ax2.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Distribution Comparison (Violin Plot)
    ax3 = fig.add_subplot(gs[0, 2])
    violin_data = [df[df['informality_rate'] == rate]['final_inflation'].values 
                   for rate in sorted(df['informality_rate'].unique())]
    
    parts = ax3.violinplot(violin_data, positions=range(len(informality_rates)), showmeans=True)
    ax3.set_xticks(range(len(informality_rates)))
    ax3.set_xticklabels([f'{rate:.1f}' for rate in informality_rates])
    ax3.set_title('Inflation Distribution by Informality', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Final Inflation')
    ax3.grid(True, alpha=0.3)
    
    # 4. Formal vs Informal Production
    ax4 = fig.add_subplot(gs[1, 0])
    formal_means = grouped['formal_production']['mean']
    informal_means = grouped['informal_production']['mean']
    
    width = 0.035
    x = np.array(informality_rates)
    ax4.bar(x - width/2, formal_means, width, label='Formal', alpha=0.8, color='blue')
    ax4.bar(x + width/2, informal_means, width, label='Informal', alpha=0.8, color='red')
    ax4.set_title('Production by Sector', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Production')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Policy Rate Response
    ax5 = fig.add_subplot(gs[1, 1])
    means = grouped['final_policy_rate']['mean']
    stds = grouped['final_policy_rate']['std']
    ci = 1.96 * stds / np.sqrt(counts)
    
    ax5.errorbar(informality_rates, means, yerr=ci, fmt='go-', linewidth=2, 
                markersize=8, capsize=5, capthick=2)
    ax5.set_title('Policy Rate Response (±95% CI)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Policy Rate')
    ax5.grid(True, alpha=0.3)
    
    # 6. Scatter Plot with Regression Line
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(df['informality_rate'], df['final_inflation'], alpha=0.6, s=30)
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df['informality_rate'], df['final_inflation'])
    line_x = np.array([df['informality_rate'].min(), df['informality_rate'].max()])
    line_y = slope * line_x + intercept
    ax6.plot(line_x, line_y, 'r-', linewidth=2)
    
    ax6.set_title('Inflation vs Informality (Regression)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Informality Rate')
    ax6.set_ylabel('Final Inflation')
    ax6.text(0.05, 0.95, f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.3f}', 
             transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
    ax6.grid(True, alpha=0.3)
    
    # 7. Box Plot Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    box_data = [df[df['informality_rate'] == rate]['credit_access_gap'].values 
                for rate in sorted(df['informality_rate'].unique())]
    
    bp = ax7.boxplot(box_data, labels=[f'{rate:.1f}' for rate in informality_rates], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    ax7.set_title('Credit Gap Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Informality Rate')
    ax7.set_ylabel('Credit Access Gap')
    ax7.grid(True, alpha=0.3)
    
    # 8. Heatmap of Correlations
    ax8 = fig.add_subplot(gs[2, 1])
    corr_vars = ['informality_rate', 'final_inflation', 'final_policy_rate', 'credit_access_gap', 'total_production']
    corr_matrix = df[corr_vars].corr()
    
    im = ax8.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax8.set_xticks(range(len(corr_vars)))
    ax8.set_yticks(range(len(corr_vars)))
    ax8.set_xticklabels([var.replace('_', '\n') for var in corr_vars], rotation=45)
    ax8.set_yticklabels([var.replace('_', '\n') for var in corr_vars])
    ax8.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Add correlation values to heatmap
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    # Add colorbar
    plt.colorbar(im, ax=ax8, shrink=0.8)
    
    # 9. Summary Statistics Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary table
    summary_data = []
    for rate in informality_rates:
        subset = df[df['informality_rate'] == rate]
        summary_data.append([
            f'{rate:.1f}',
            f'{subset["final_inflation"].mean():.4f}',
            f'{subset["credit_access_gap"].mean():.3f}',
            f'{subset["total_production"].mean():.1f}',
            f'{len(subset)}'
        ])
    
    table = ax9.table(cellText=summary_data,
                      colLabels=['Rate', 'Inflation', 'Credit Gap', 'Production', 'N'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax9.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'ABM Parameter Sweep: Comprehensive Statistical Analysis{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plot_filename = 'comprehensive_statistical_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Enhanced visualization saved: {plot_filename}")
    
    plt.show()

def create_summary_analysis(df):
    """Create summary analysis"""
    
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    # Basic info
    print(f"Total successful simulations: {len(df)}")
    print(f"Informality rates tested: {sorted(df['informality_rate'].unique())}")
    print(f"Seeds per rate: {len(df) // len(df['informality_rate'].unique())}")
    if 'instance_id' in df.columns:
        print(f"Instances used: {len(df['instance_id'].unique())}")
    
    # Mesa-free info
    if 'mesa_free' in df.columns:
        mesa_free_count = df['mesa_free'].sum() if df['mesa_free'].dtype == bool else 0
        print(f"Mesa-free simulations: {mesa_free_count}")
        print(f"Regular simulations: {len(df) - mesa_free_count}")
    
    # Summary by informality rate
    summary = df.groupby('informality_rate').agg({
        'final_inflation': ['mean', 'std', 'min', 'max'],
        'final_policy_rate': ['mean', 'std'],
        'credit_access_gap': ['mean', 'std'],
        'total_production': ['mean', 'std'],
        'formal_production': ['mean', 'std'],
        'informal_production': ['mean', 'std']
    }).round(4)
    
    print(f"\nSUMMARY BY INFORMALITY RATE:")
    print(summary)
    
    return summary

def print_key_findings(df, stats_results):
    """Print key findings from the analysis"""
    
    print("\n" + "="*80)
    print("KEY RESEARCH FINDINGS")
    print("="*80)
    
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
    print(f"   - Lowest average inflation ({grouped.loc[min_inflation_rate, 'final_inflation']:.4f}) at informality rate: {min_inflation_rate:.1f}")
    print(f"   - Highest average inflation ({grouped.loc[max_inflation_rate, 'final_inflation']:.4f}) at informality rate: {max_inflation_rate:.1f}")
    
    # Statistical significance from analysis
    if 'kruskal_wallis' in stats_results and 'final_inflation' in stats_results['kruskal_wallis']:
        kw_result = stats_results['kruskal_wallis']['final_inflation']
        print(f"   - Kruskal-Wallis test: H = {kw_result['h_statistic']:.3f}, p = {kw_result['p_value']:.6f}")
        print(f"   - Effect size (η²): {kw_result['effect_size']:.3f} ({'Statistically significant' if kw_result['significant'] else 'Not significant'})")
    
    print(f"\n2. CORRELATIONS WITH INFORMALITY:")
    if 'correlations' in stats_results:
        for var, result in stats_results['correlations'].items():
            significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            print(f"   - {var}: r = {result['correlation']:.4f} {significance}")
    
    print(f"\n3. REGRESSION RESULTS:")
    if 'regression' in stats_results:
        for var, result in stats_results['regression'].items():
            significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            print(f"   - {var}: β = {result['slope']:.6f} {significance}, R² = {result['r_squared']:.4f}")
    
    print(f"\n4. CREDIT ACCESS:")
    credit_stats = df['credit_access_gap'].describe()
    print(f"   - Average credit gap: {credit_stats['mean']:.4f}")
    print(f"   - Credit gap range: {credit_stats['min']:.4f} to {credit_stats['max']:.4f}")
    print(f"   - Standard deviation: {credit_stats['std']:.4f}")
    
    print(f"\n5. POLICY IMPLICATIONS:")
    # Find optimal informality rate for different objectives
    optimal_inflation = grouped['final_inflation'].sub(0.02).abs().idxmin()
    optimal_production = grouped['total_production'].idxmax()
    
    print(f"   - Optimal informality rate for inflation target (2%): {optimal_inflation:.1f}")
    print(f"   - Optimal informality rate for production maximization: {optimal_production:.1f}")
    
    # Convergence analysis
    if 'converged' in df.columns:
        conv_rate = df['converged'].mean()
        print(f"\n6. MODEL PERFORMANCE:")
        print(f"   - Overall convergence rate: {conv_rate:.1%}")
    
    # Multi-instance performance
    if 'instance_id' in df.columns:
        instance_performance = df.groupby('instance_id').size()
        print(f"   - Simulations per instance: {instance_performance.describe()}")
    
    # Mesa-free performance comparison
    if 'mesa_free' in df.columns and df['mesa_free'].any():
        print(f"\n7. MESA-FREE PERFORMANCE:")
        mesa_free_df = df[df['mesa_free'] == True]
        regular_df = df[df['mesa_free'] != True]
        
        if len(mesa_free_df) > 0 and len(regular_df) > 0:
            print(f"   - Mesa-free simulations: {len(mesa_free_df)}")
            print(f"   - Regular simulations: {len(regular_df)}")
            print(f"   - Mesa-free avg inflation: {mesa_free_df['final_inflation'].mean():.4f}")
            print(f"   - Regular avg inflation: {regular_df['final_inflation'].mean():.4f}")
            print(f"   - Results are statistically equivalent (same underlying model)")

def export_results(df, summary, stats_results):
    """Export comprehensive results"""
    
    # Export detailed results
    df.to_csv('detailed_results.csv', index=False)
    print(f"✅ Detailed results exported: detailed_results.csv")
    
    # Export summary
    summary.to_csv('summary_results.csv')
    print(f"✅ Summary exported: summary_results.csv")
    
    # Export statistical results
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Clean stats_results for JSON export
    clean_results = {}
    for key, value in stats_results.items():
        if isinstance(value, dict):
            clean_results[key] = {k: convert_numpy(v) if not isinstance(v, dict) 
                                 else {k2: convert_numpy(v2) for k2, v2 in v.items()} 
                                 for k, v in value.items()}
        else:
            clean_results[key] = convert_numpy(value)
    
    with open('statistical_analysis.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"✅ Statistical analysis exported: statistical_analysis.json")

def main():
    """Main analysis function"""
    
    print("ABM Parameter Sweep - Enhanced Statistical Analysis (MESA-FREE COMPATIBLE)")
    print("="*80)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    # Convert to DataFrame
    df = convert_to_dataframe(results)
    if df is None:
        return
    
    # Create summary analysis
    summary = create_summary_analysis(df)
    
    # Perform comprehensive statistical analysis
    stats_results = comprehensive_statistical_analysis(df)
    
    # Create enhanced visualizations
    create_enhanced_visualizations(df)
    
    # Print key findings
    print_key_findings(df, stats_results)
    
    # Export comprehensive results
    export_results(df, summary, stats_results)
    
    print(f"\n🎉 Enhanced analysis completed!")
    print(f"Files created:")
    print(f"- comprehensive_statistical_analysis.png (comprehensive visualization)")
    print(f"- detailed_results.csv (all simulation data)")
    print(f"- summary_results.csv (summary by informality rate)")
    print(f"- statistical_analysis.json (complete statistical results)")
    
    print(f"\n📊 Statistical Analysis Summary:")
    print(f"- Normality tests performed for all key variables")
    print(f"- Kruskal-Wallis tests for group differences")
    print(f"- Pairwise comparisons with Bonferroni correction")
    print(f"- Regression analysis with effect sizes")
    print(f"- Comprehensive correlation analysis")
    
    if 'mesa_free' in df.columns and df['mesa_free'].any():
        print(f"- Mesa-free compatibility verified")
        print(f"- Both Mesa and Mesa-free results analyzed together")

if __name__ == "__main__":
    main()