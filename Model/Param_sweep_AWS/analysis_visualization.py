import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib styling")

class ParameterSweepAnalyzer:
    """
    Comprehensive analysis and visualization of parameter sweep results
    """
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Remove failed simulations if error column exists
        if 'error' in self.results_df.columns:
            self.results_df = self.results_df[self.results_df['error'].isna()]
        
        # Extract parameter columns if they're nested
        if 'parameters.informality_rate' in self.results_df.columns:
            self.results_df['informality_rate'] = self.results_df['parameters.informality_rate']
            self.results_df['seed'] = self.results_df['parameters.seed']
        elif 'parameters' in self.results_df.columns:
            # Handle case where parameters are in a single column
            import ast
            for idx, row in self.results_df.iterrows():
                if isinstance(row['parameters'], str):
                    params = ast.literal_eval(row['parameters'])
                elif isinstance(row['parameters'], dict):
                    params = row['parameters']
                else:
                    continue
                self.results_df.at[idx, 'informality_rate'] = params.get('informality_rate')
                self.results_df.at[idx, 'seed'] = params.get('seed')
        
        print(f"Cleaned dataset: {len(self.results_df)} successful simulations")
        
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics"""
        
        # Group by informality rate
        summary = self.results_df.groupby('informality_rate').agg({
            # Convergence and performance
            'converged': ['mean', 'count'],
            'run_time': ['mean', 'std'],
            'final_step': ['mean', 'std'],
            
            # Economic indicators
            'final_inflation': ['mean', 'std', 'min', 'max'],
            'final_policy_rate': ['mean', 'std', 'min', 'max'],
            'final_output_gap': ['mean', 'std', 'min', 'max'],
            'avg_lending_rate': ['mean', 'std'],
            
            # Informality metrics
            'informal_firm_pct': ['mean', 'std'],
            'informal_consumer_pct': ['mean', 'std'],
            'credit_access_gap': ['mean', 'std', 'min', 'max'],
            
            # Banking and production
            'total_formal_loans': ['mean', 'std'],
            'total_informal_loans': ['mean', 'std'],
            'formal_production': ['mean', 'std'],
            'informal_production': ['mean', 'std'],
            'total_production': ['mean', 'std'],
            'total_consumption': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = [f'{col[0]}_{col[1]}' for col in summary.columns]
        summary = summary.reset_index()
        
        return summary
    
    def plot_main_results(self, save_path: Optional[str] = None):
        """Create comprehensive visualization of main results"""
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Parameter Sweep Results: Impact of Informality Rate', fontsize=16, fontweight='bold')
        
        # Calculate means and confidence intervals
        summary_stats = self.results_df.groupby('informality_rate').agg({
            'final_inflation': ['mean', 'std', 'count'],
            'final_policy_rate': ['mean', 'std', 'count'],
            'credit_access_gap': ['mean', 'std', 'count'],
            'final_output_gap': ['mean', 'std', 'count'],
            'total_production': ['mean', 'std', 'count'],
            'converged': ['mean', 'count'],
            'avg_lending_rate': ['mean', 'std', 'count'],
            'total_formal_loans': ['mean', 'std', 'count'],
            'total_informal_loans': ['mean', 'std', 'count']
        })
        
        informality_rates = summary_stats.index
        
        # 1. Final Inflation
        ax = axes[0, 0]
        means = summary_stats['final_inflation']['mean']
        stds = summary_stats['final_inflation']['std']
        counts = summary_stats['final_inflation']['count']
        ci = 1.96 * stds / np.sqrt(counts)  # 95% confidence interval
        
        ax.plot(informality_rates, means, 'bo-', linewidth=2, markersize=8)
        ax.fill_between(informality_rates, means - ci, means + ci, alpha=0.3)
        ax.axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='Target (2%)')
        ax.set_title('Final Inflation Rate')
        ax.set_ylabel('Inflation Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Policy Rate
        ax = axes[0, 1]
        means = summary_stats['final_policy_rate']['mean']
        stds = summary_stats['final_policy_rate']['std']
        ci = 1.96 * stds / np.sqrt(counts)
        
        ax.plot(informality_rates, means, 'go-', linewidth=2, markersize=8)
        ax.fill_between(informality_rates, means - ci, means + ci, alpha=0.3, color='green')
        ax.set_title('Final Policy Rate')
        ax.set_ylabel('Policy Rate')
        ax.grid(True, alpha=0.3)
        
        # 3. Credit Access Gap
        ax = axes[0, 2]
        means = summary_stats['credit_access_gap']['mean']
        stds = summary_stats['credit_access_gap']['std']
        ci = 1.96 * stds / np.sqrt(counts)
        
        ax.plot(informality_rates, means, 'ro-', linewidth=2, markersize=8)
        ax.fill_between(informality_rates, means - ci, means + ci, alpha=0.3, color='red')
        ax.set_title('Credit Access Gap')
        ax.set_ylabel('Gap (Formal - Informal)')
        ax.grid(True, alpha=0.3)
        
        # 4. Output Gap
        ax = axes[1, 0]
        means = summary_stats['final_output_gap']['mean']
        stds = summary_stats['final_output_gap']['std']
        ci = 1.96 * stds / np.sqrt(counts)
        
        ax.plot(informality_rates, means, 'mo-', linewidth=2, markersize=8)
        ax.fill_between(informality_rates, means - ci, means + ci, alpha=0.3, color='purple')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax.set_title('Final Output Gap')
        ax.set_ylabel('Output Gap')
        ax.grid(True, alpha=0.3)
        
        # 5. Total Production
        ax = axes[1, 1]
        means = summary_stats['total_production']['mean']
        stds = summary_stats['total_production']['std']
        ci = 1.96 * stds / np.sqrt(counts)
        
        ax.plot(informality_rates, means, 'co-', linewidth=2, markersize=8)
        ax.fill_between(informality_rates, means - ci, means + ci, alpha=0.3, color='cyan')
        ax.set_title('Total Production')
        ax.set_ylabel('Production')
        ax.grid(True, alpha=0.3)
        
        # 6. Convergence Rate
        ax = axes[1, 2]
        conv_rates = summary_stats['converged']['mean']
        ax.bar(informality_rates, conv_rates, alpha=0.7, color='orange')
        ax.set_title('Convergence Rate')
        ax.set_ylabel('Proportion Converged')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 7. Lending Rates
        ax = axes[2, 0]
        means = summary_stats['avg_lending_rate']['mean']
        stds = summary_stats['avg_lending_rate']['std']
        ci = 1.96 * stds / np.sqrt(counts)
        
        ax.plot(informality_rates, means, 'yo-', linewidth=2, markersize=8)
        ax.fill_between(informality_rates, means - ci, means + ci, alpha=0.3, color='yellow')
        ax.set_title('Average Lending Rate')
        ax.set_ylabel('Lending Rate')
        ax.grid(True, alpha=0.3)
        
        # 8. Formal vs Informal Loans
        ax = axes[2, 1]
        formal_means = summary_stats['total_formal_loans']['mean']
        informal_means = summary_stats['total_informal_loans']['mean']
        
        width = 0.035
        x = np.array(informality_rates)
        ax.bar(x - width/2, formal_means, width, label='Formal Loans', alpha=0.7, color='blue')
        ax.bar(x + width/2, informal_means, width, label='Informal Loans', alpha=0.7, color='red')
        ax.set_title('Total Loans by Sector')
        ax.set_ylabel('Loan Amount')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Distribution of Final Inflation (boxplot)
        ax = axes[2, 2]
        inflation_data = [self.results_df[self.results_df['informality_rate'] == rate]['final_inflation'].values 
                         for rate in sorted(self.results_df['informality_rate'].unique())]
        
        box_plot = ax.boxplot(inflation_data, labels=[f'{rate:.1f}' for rate in sorted(self.results_df['informality_rate'].unique())])
        ax.axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='Target')
        ax.set_title('Distribution of Final Inflation')
        ax.set_ylabel('Final Inflation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add common x-label
        for ax in axes[2, :]:
            ax.set_xlabel('Informality Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Main results plot saved to: {save_path}")
        
        plt.show()
        
    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Final Inflation vs Informality', 'Credit Access Gap', 'Production by Sector',
                           'Convergence Rate', 'Lending Rates', 'Economic Stability'],
            specs=[[{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calculate summary statistics
        summary = self.results_df.groupby('informality_rate').agg({
            'final_inflation': ['mean', 'std'],
            'credit_access_gap': ['mean', 'std'],
            'formal_production': 'mean',
            'informal_production': 'mean',
            'converged': 'mean',
            'avg_lending_rate': ['mean', 'std'],
            'final_policy_rate': ['mean', 'std']
        })
        
        informality_rates = summary.index
        
        # 1. Final Inflation
        fig.add_trace(
            go.Scatter(
                x=informality_rates,
                y=summary['final_inflation']['mean'],
                mode='lines+markers',
                name='Final Inflation',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # 2. Credit Access Gap
        fig.add_trace(
            go.Scatter(
                x=informality_rates,
                y=summary['credit_access_gap']['mean'],
                mode='lines+markers',
                name='Credit Gap',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # 3. Production by Sector
        fig.add_trace(
            go.Scatter(
                x=informality_rates,
                y=summary['formal_production']['mean'],
                mode='lines+markers',
                name='Formal Production',
                line=dict(color='blue', width=3)
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=informality_rates,
                y=summary['informal_production']['mean'],
                mode='lines+markers',
                name='Informal Production',
                line=dict(color='red', width=3)
            ),
            row=1, col=3
        )
        
        # 4. Convergence Rate
        fig.add_trace(
            go.Bar(
                x=informality_rates,
                y=summary['converged']['mean'],
                name='Convergence Rate',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 5. Lending Rates
        fig.add_trace(
            go.Scatter(
                x=informality_rates,
                y=summary['avg_lending_rate']['mean'],
                mode='lines+markers',
                name='Lending Rate',
                line=dict(color='green', width=3)
            ),
            row=2, col=2
        )
        
        # 6. Economic Stability (Policy Rate)
        fig.add_trace(
            go.Scatter(
                x=informality_rates,
                y=summary['final_policy_rate']['mean'],
                mode='lines+markers',
                name='Policy Rate',
                line=dict(color='navy', width=3)
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Parameter Sweep Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update x-axis labels
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(title_text="Informality Rate", row=i, col=j)
        
        return fig
    
    def statistical_analysis(self) -> Dict:
        """Perform statistical analysis of the results"""
        
        results = {}
        
        # 1. Correlation analysis
        numeric_cols = ['informality_rate', 'final_inflation', 'final_policy_rate', 
                       'credit_access_gap', 'final_output_gap', 'total_production',
                       'avg_lending_rate', 'formal_production', 'informal_production']
        
        correlation_matrix = self.results_df[numeric_cols].corr()
        results['correlations'] = correlation_matrix
        
        # 2. ANOVA for informality rate effects
        informality_groups = [group['final_inflation'].values for name, group in self.results_df.groupby('informality_rate')]
        f_stat, p_value = stats.f_oneway(*informality_groups)
        
        results['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # 3. Regression analysis: Informality rate vs key outcomes
        from scipy.stats import linregress
        
        # Inflation
        slope, intercept, r_value, p_value, std_err = linregress(
            self.results_df['informality_rate'], 
            self.results_df['final_inflation']
        )
        
        results['inflation_regression'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value
        }
        
        # Credit gap
        slope, intercept, r_value, p_value, std_err = linregress(
            self.results_df['informality_rate'], 
            self.results_df['credit_access_gap']
        )
        
        results['credit_gap_regression'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value
        }
        
        return results
    
    def export_results(self, filename_prefix: str = "analysis_results"):
        """Export all analysis results"""
        
        # Summary statistics
        summary = self.generate_summary_statistics()
        summary.to_csv(f"{filename_prefix}_summary.csv", index=False)
        
        # Statistical analysis
        stats_results = self.statistical_analysis()
        
        # Save correlation matrix
        stats_results['correlations'].to_csv(f"{filename_prefix}_correlations.csv")
        
        # Save regression results
        with open(f"{filename_prefix}_statistical_tests.json", 'w') as f:
            # Remove non-serializable items
            export_stats = {k: v for k, v in stats_results.items() if k != 'correlations'}
            json.dump(export_stats, f, indent=2)
        
        print(f"Analysis results exported with prefix: {filename_prefix}")
        
        return summary, stats_results

# Example usage function
def analyze_sweep_results(results_path: str, detailed_results_path: Optional[str] = None):
    """
    Complete analysis pipeline for parameter sweep results
    """
    
    print("Loading results...")
    
    # Load main results
    if results_path.endswith('.csv'):
        results_df = pd.read_csv(results_path)
    elif results_path.endswith('.pkl'):
        with open(results_path, 'rb') as f:
            results_df = pickle.load(f)
    else:
        raise ValueError("Results file must be .csv or .pkl")
    
    # Initialize analyzer
    analyzer = ParameterSweepAnalyzer(results_df)
    
    print(f"Analyzing {len(analyzer.results_df)} successful simulations...")
    
    # Generate summary statistics
    print("\n=== GENERATING SUMMARY STATISTICS ===")
    summary = analyzer.generate_summary_statistics()
    print(summary)
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    analyzer.plot_main_results(save_path="parameter_sweep_results.png")
    
    # Create interactive dashboard
    print("\n=== CREATING INTERACTIVE DASHBOARD ===")
    interactive_fig = analyzer.create_interactive_dashboard()
    interactive_fig.write_html("interactive_dashboard.html")
    interactive_fig.show()
    
    # Statistical analysis
    print("\n=== STATISTICAL ANALYSIS ===")
    stats_results = analyzer.statistical_analysis()
    
    print("ANOVA Results:")
    print(f"F-statistic: {stats_results['anova']['f_statistic']:.4f}")
    print(f"P-value: {stats_results['anova']['p_value']:.6f}")
    print(f"Significant: {stats_results['anova']['significant']}")
    
    print("\nInflation-Informality Regression:")
    inf_reg = stats_results['inflation_regression']
    print(f"Slope: {inf_reg['slope']:.6f}")
    print(f"R-squared: {inf_reg['r_squared']:.4f}")
    print(f"P-value: {inf_reg['p_value']:.6f}")
    
    print("\nCredit Gap-Informality Regression:")
    credit_reg = stats_results['credit_gap_regression']
    print(f"Slope: {credit_reg['slope']:.6f}")
    print(f"R-squared: {credit_reg['r_squared']:.4f}")
    print(f"P-value: {credit_reg['p_value']:.6f}")
    
    # Export results
    print("\n=== EXPORTING RESULTS ===")
    analyzer.export_results("complete_analysis")
    
    # Key insights
    print("\n=== KEY INSIGHTS ===")
    print_key_insights(summary, stats_results)
    
    return analyzer, summary, stats_results

def print_key_insights(summary: pd.DataFrame, stats_results: Dict):
    """Print key insights from the analysis"""
    
    print("1. INFLATION DYNAMICS:")
    min_inflation_rate = summary.loc[summary['final_inflation_mean'].idxmin(), 'informality_rate']
    max_inflation_rate = summary.loc[summary['final_inflation_mean'].idxmax(), 'informality_rate']
    print(f"   - Lowest inflation at informality rate: {min_inflation_rate:.1f}")
    print(f"   - Highest inflation at informality rate: {max_inflation_rate:.1f}")
    
    # Check if inflation generally increases with informality
    slope = stats_results['inflation_regression']['slope']
    if slope > 0:
        print(f"   - Inflation INCREASES with informality (slope: {slope:.6f})")
    else:
        print(f"   - Inflation DECREASES with informality (slope: {slope:.6f})")
    
    print("\n2. CREDIT ACCESS:")
    avg_credit_gap = summary['credit_access_gap_mean'].mean()
    max_credit_gap = summary['credit_access_gap_mean'].max()
    print(f"   - Average credit access gap: {avg_credit_gap:.3f}")
    print(f"   - Maximum credit access gap: {max_credit_gap:.3f}")
    
    credit_slope = stats_results['credit_gap_regression']['slope']
    if credit_slope > 0:
        print(f"   - Credit gap INCREASES with informality (slope: {credit_slope:.3f})")
    else:
        print(f"   - Credit gap DECREASES with informality (slope: {credit_slope:.3f})")
    
    print("\n3. CONVERGENCE:")
    avg_convergence = summary['converged_mean'].mean()
    min_convergence_rate = summary.loc[summary['converged_mean'].idxmin(), 'informality_rate']
    print(f"   - Overall convergence rate: {avg_convergence:.1%}")
    print(f"   - Lowest convergence at informality rate: {min_convergence_rate:.1f}")
    
    print("\n4. PRODUCTION:")
    total_prod_corr = summary[['informality_rate', 'total_production_mean']].corr().iloc[0,1]
    if total_prod_corr > 0:
        print(f"   - Total production INCREASES with informality (corr: {total_prod_corr:.3f})")
    else:
        print(f"   - Total production DECREASES with informality (corr: {total_prod_corr:.3f})")
    
    print("\n5. POLICY IMPLICATIONS:")
    # Find optimal informality rate for different objectives
    optimal_inflation = summary.loc[
        (summary['final_inflation_mean'] - 0.02).abs().idxmin(), 'informality_rate'
    ]
    optimal_production = summary.loc[summary['total_production_mean'].idxmax(), 'informality_rate']
    
    print(f"   - Optimal informality rate for inflation target: {optimal_inflation:.1f}")
    print(f"   - Optimal informality rate for production: {optimal_production:.1f}")


# Complete example of how to use the system
if __name__ == "__main__":
    
    # Example: Load and analyze existing results
    try:
        # Replace with your actual results file
        results_file = "sweep_results/parameter_sweep_results_20241201_120000.csv"
        analyzer, summary, stats = analyze_sweep_results(results_file)
        
    except FileNotFoundError:
        print("No existing results found. Run the parameter sweep first.")
        print("Example command:")
        print("python parameter_sweep_runner.py")