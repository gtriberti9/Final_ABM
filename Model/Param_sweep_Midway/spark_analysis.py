#!/usr/bin/env python3
"""
Scalable Analysis of ABM Parameter Sweep Results using Apache Spark

This script processes large datasets from ABM parameter sweeps using Apache Spark
for distributed computing. It can handle millions of simulation results efficiently.

What this does:
1. Loads simulation results from CSV files into Spark
2. Performs statistical analysis across different informality rates
3. Creates comprehensive visualizations
4. Generates detailed reports with economic insights

When to use this:
- When you have thousands or millions of simulation results
- When regular pandas analysis is too slow or runs out of memory
- When you want to leverage cluster computing for analysis

Requirements:
- Apache Spark installed and configured
- PySpark Python package
- Standard data science packages (pandas, matplotlib, seaborn)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Bucketizer
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime


def create_spark_session():
    """
    Create and configure a Spark session for processing our data
    
    Spark is a distributed computing framework that can process large datasets
    across multiple machines (or cores). This function sets up Spark with
    optimizations for our specific use case.
    
    Returns:
        SparkSession: Configured Spark session ready for data processing
    """
    return SparkSession.builder \
        .appName("ABM_Parameter_Sweep_Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()


def load_and_prepare_data(spark, csv_file_path):
    """
    Load CSV data into Spark and prepare it for analysis
    
    This function:
    1. Defines the data schema (column types) for better performance
    2. Loads the CSV file into a Spark DataFrame
    3. Creates additional calculated columns for analysis
    
    Args:
        spark: SparkSession object
        csv_file_path: Path to the CSV file with simulation results
    
    Returns:
        DataFrame: Spark DataFrame with loaded and prepared data
    """
    
    # Define the schema (data types) for each column
    # This is much faster than letting Spark infer types automatically
    # Each StructField defines: (column_name, data_type, nullable)
    schema = StructType([
        # Basic simulation identifiers
        StructField("sim_id", IntegerType(), True),
        StructField("informality_rate", DoubleType(), True),
        StructField("seed", IntegerType(), True),
        StructField("steps_run", IntegerType(), True),
        
        # Convergence metrics
        StructField("converged", BooleanType(), True),
        StructField("convergence_time", IntegerType(), True),
        
        # Final economic state
        StructField("final_inflation", DoubleType(), True),
        StructField("final_policy_rate", DoubleType(), True),
        StructField("final_output_gap", DoubleType(), True),
        StructField("avg_lending_rate", DoubleType(), True),
        
        # Volatility measures
        StructField("inflation_volatility", DoubleType(), True),
        StructField("policy_rate_volatility", DoubleType(), True),
        
        # Agent counts by sector
        StructField("formal_firms_count", IntegerType(), True),
        StructField("informal_firms_count", IntegerType(), True),
        StructField("formal_consumers_count", IntegerType(), True),
        StructField("informal_consumers_count", IntegerType(), True),
        
        # Production and consumption by sector
        StructField("formal_production", DoubleType(), True),
        StructField("informal_production", DoubleType(), True),
        StructField("total_production", DoubleType(), True),
        StructField("production_ratio_formal", DoubleType(), True),
        StructField("formal_consumption", DoubleType(), True),
        StructField("informal_consumption", DoubleType(), True),
        StructField("total_consumption", DoubleType(), True),
        StructField("consumption_ratio_formal", DoubleType(), True),
        
        # Credit and banking metrics
        StructField("total_formal_loans", DoubleType(), True),
        StructField("total_informal_loans", DoubleType(), True),
        StructField("credit_access_gap", DoubleType(), True),
        StructField("formal_credit_ratio", DoubleType(), True),
        
        # Average firm-level metrics by sector
        StructField("avg_formal_firm_production", DoubleType(), True),
        StructField("avg_informal_firm_production", DoubleType(), True),
        StructField("avg_formal_firm_price", DoubleType(), True),
        StructField("avg_informal_firm_price", DoubleType(), True),
        StructField("avg_formal_credit_access", DoubleType(), True),
        StructField("avg_informal_credit_access", DoubleType(), True)
    ])
    
    # Load the CSV file using the defined schema
    print(f"Loading data from {csv_file_path}...")
    df = spark.read.csv(csv_file_path, header=True, schema=schema)
    
    # Add derived columns for analysis
    print("Creating derived columns...")
    
    # Categorize informality rates into groups for easier analysis
    df = df.withColumn("informality_category", 
                      when(col("informality_rate") <= 0.2, "Low")      # 0-20%
                      .when(col("informality_rate") <= 0.5, "Medium")  # 20-50%
                      .otherwise("High"))                              # 50%+
    
    # Convergence efficiency: how quickly did simulations converge?
    # Only defined for simulations that actually converged
    df = df.withColumn("convergence_efficiency", 
                      when(col("converged"), col("convergence_time"))
                      .otherwise(lit(None)))
    
    # Policy effectiveness: how far is final inflation from the 2% target?
    # Lower values = better policy effectiveness
    df = df.withColumn("policy_effectiveness", 
                      abs(col("final_inflation") - 0.02))  # Distance from 2% target
    
    # Economic stability: combined measure of inflation and policy rate volatility
    # Higher values = less stable economy
    df = df.withColumn("economic_stability", 
                      col("inflation_volatility") + col("policy_rate_volatility"))
    
    # Create measures of gaps between formal and informal sectors
    
    # Production gap: difference in average productivity between sectors
    df = df.withColumn("production_gap", 
                      col("avg_formal_firm_production") - col("avg_informal_firm_production"))
    
    # Price gap: difference in average prices between sectors
    df = df.withColumn("price_gap", 
                      col("avg_formal_firm_price") - col("avg_informal_firm_price"))
    
    print(f"Data preparation complete. Loaded {df.count()} rows.")
    return df


def comprehensive_analysis(df, output_dir):
    """
    Perform comprehensive statistical analysis of the parameter sweep results
    
    This function computes various summary statistics and analyzes patterns
    in the data across different dimensions (informality rates, regimes, etc.)
    
    Args:
        df: Spark DataFrame with simulation results
        output_dir: Directory where analysis results will be saved
        
    Returns:
        tuple: (analysis_results_dict, by_informality_pandas_df, correlation_matrix)
    """
    
    analysis_results = {}
    print("Performing comprehensive analysis...")
    
    # 1. OVERALL STATISTICS
    # Compute summary statistics across all simulations
    print("1. Computing overall statistics...")
    
    total_sims = df.count()  # Total number of simulations
    convergence_rate = df.filter(col("converged")).count() / total_sims  # Fraction that converged
    
    # Compute summary statistics for key economic variables
    overall_stats = df.agg(
        mean("final_inflation").alias("avg_inflation"),
        stddev("final_inflation").alias("std_inflation"),
        mean("final_policy_rate").alias("avg_policy_rate"),
        stddev("final_policy_rate").alias("std_policy_rate"),
        mean("inflation_volatility").alias("avg_inflation_volatility"),
        mean("credit_access_gap").alias("avg_credit_gap"),
        mean("production_ratio_formal").alias("avg_formal_production_ratio")
    ).collect()[0]  # collect()[0] gets the single row of results
    
    # Store overall statistics
    analysis_results['overall'] = {
        'total_simulations': total_sims,
        'convergence_rate': convergence_rate,
        'avg_inflation': overall_stats['avg_inflation'],
        'std_inflation': overall_stats['std_inflation'],
        'avg_policy_rate': overall_stats['avg_policy_rate'],
        'std_policy_rate': overall_stats['std_policy_rate'],
        'avg_inflation_volatility': overall_stats['avg_inflation_volatility'],
        'avg_credit_gap': overall_stats['avg_credit_gap'],
        'avg_formal_production_ratio': overall_stats['avg_formal_production_ratio']
    }
    
    # 2. ANALYSIS BY INFORMALITY RATE
    # This is the main analysis: how do outcomes vary with informality?
    print("2. Analyzing by informality rate...")
    
    by_informality = df.groupBy("informality_rate").agg(
        count("*").alias("n_simulations"),  # Number of simulations per rate
        
        # Convergence statistics
        mean(col("converged").cast("double")).alias("convergence_rate"),  # Cast boolean to double for averaging
        mean("convergence_time").alias("avg_convergence_time"),
        
        # Inflation performance
        mean("final_inflation").alias("avg_inflation"),
        stddev("final_inflation").alias("std_inflation"),
        mean("inflation_volatility").alias("avg_inflation_volatility"),
        
        # Policy and stability metrics
        mean("policy_effectiveness").alias("avg_policy_effectiveness"),
        mean("economic_stability").alias("avg_economic_stability"),
        
        # Credit and sectoral metrics
        mean("credit_access_gap").alias("avg_credit_gap"),
        mean("production_ratio_formal").alias("avg_formal_production_ratio"),
        mean("formal_credit_ratio").alias("avg_formal_credit_ratio"),
        
        # Gaps between sectors
        mean("production_gap").alias("avg_production_gap"),
        mean("price_gap").alias("avg_price_gap")
    ).orderBy("informality_rate")  # Sort by informality rate
    
    # Convert Spark DataFrame to Pandas for easier visualization and saving
    by_informality_pd = by_informality.toPandas()
    analysis_results['by_informality'] = by_informality_pd.to_dict('records')
    
    # 3. CORRELATION ANALYSIS
    # Understand relationships between different variables
    print("3. Computing correlations...")
    
    # Select key variables for correlation analysis
    correlation_features = [
        "informality_rate", "final_inflation", "final_policy_rate", 
        "inflation_volatility", "credit_access_gap", "production_ratio_formal",
        "formal_credit_ratio", "policy_effectiveness", "economic_stability"
    ]
    
    # Convert to pandas for correlation matrix calculation
    # (Spark's correlation functions are more complex to use)
    correlation_data = df.select(*correlation_features).toPandas()
    correlation_matrix = correlation_data.corr()
    
    analysis_results['correlations'] = correlation_matrix.to_dict()
    
    # 4. REGIME ANALYSIS
    # Compare outcomes across low/medium/high informality regimes
    print("4. Analyzing by informality regime...")
    
    by_regime = df.groupBy("informality_category").agg(
        count("*").alias("n_simulations"),
        mean(col("converged").cast("double")).alias("convergence_rate"),
        mean("final_inflation").alias("avg_inflation"),
        mean("inflation_volatility").alias("avg_inflation_volatility"),
        mean("credit_access_gap").alias("avg_credit_gap"),
        mean("production_ratio_formal").alias("avg_formal_production_ratio"),
        mean("policy_effectiveness").alias("avg_policy_effectiveness")
    ).orderBy("informality_category")
    
    by_regime_pd = by_regime.toPandas()
    analysis_results['by_regime'] = by_regime_pd.to_dict('records')
    
    # 5. CONVERGENCE ANALYSIS
    # Focus only on simulations that converged - analyze convergence patterns
    print("5. Analyzing convergence patterns...")
    
    convergence_analysis = df.filter(col("converged")).groupBy("informality_rate").agg(
        count("*").alias("n_converged"),
        mean("convergence_time").alias("avg_convergence_time"),
        stddev("convergence_time").alias("std_convergence_time"),
        min("convergence_time").alias("min_convergence_time"),
        max("convergence_time").alias("max_convergence_time")
    ).orderBy("informality_rate")
    
    convergence_pd = convergence_analysis.toPandas()
    analysis_results['convergence'] = convergence_pd.to_dict('records')
    
    # 6. ECONOMIC OUTCOMES ANALYSIS
    # Analyze production, consumption, and lending patterns
    print("6. Analyzing economic outcomes...")
    
    economic_outcomes = df.groupBy("informality_rate").agg(
        mean("total_production").alias("avg_total_production"),
        mean("total_consumption").alias("avg_total_consumption"),
        mean("avg_lending_rate").alias("avg_lending_rate"),
        mean("total_formal_loans").alias("avg_formal_loans"),
        mean("total_informal_loans").alias("avg_informal_loans"),
        
        # Calculate formal loan share
        (mean("total_formal_loans") / 
         (mean("total_formal_loans") + mean("total_informal_loans"))).alias("formal_loan_share")
    ).orderBy("informality_rate")
    
    economic_pd = economic_outcomes.toPandas()
    analysis_results['economic_outcomes'] = economic_pd.to_dict('records')
    
    # Save all analysis results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = os.path.join(output_dir, f'analysis_results_{timestamp}.json')
    
    with open(analysis_file, 'w') as f:
        # Custom function to handle numpy types in JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(analysis_results, f, indent=2, default=convert_numpy)
    
    print(f"Analysis results saved to {analysis_file}")
    
    return analysis_results, by_informality_pd, correlation_matrix


def create_visualizations(by_informality_pd, correlation_matrix, output_dir):
    """
    Create comprehensive visualizations showing the impact of informality
    
    This function creates multiple plots to visualize:
    - How informality affects monetary policy effectiveness
    - Economic stability and volatility patterns
    - Sectoral differences and gaps
    - Correlations between variables
    
    Args:
        by_informality_pd: Pandas DataFrame with results grouped by informality rate
        correlation_matrix: Correlation matrix of key variables
        output_dir: Directory to save plot files
        
    Returns:
        tuple: (main_plot_file, correlation_plot_file, detailed_plot_file)
    """
    
    print("Creating visualizations...")
    
    # Set up plotting style for better-looking charts
    plt.style.use('default')
    sns.set_palette("husl")  # Use attractive color palette
    
    # Create a large figure with 9 subplots (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('ABM Parameter Sweep Analysis: Impact of Informality Rate', 
                 fontsize=16, fontweight='bold')
    
    # PLOT 1: Convergence Rate vs Informality
    # Shows how often monetary policy successfully converges to target
    axes[0, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['convergence_rate'], 
                    'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Convergence Rate by Informality Rate')
    axes[0, 0].set_xlabel('Informality Rate')
    axes[0, 0].set_ylabel('Convergence Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PLOT 2: Final Inflation vs Informality
    # Shows how close inflation gets to the 2% target
    axes[0, 1].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_inflation'], 
                    'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Target (2%)')
    axes[0, 1].set_title('Average Final Inflation by Informality Rate')
    axes[0, 1].set_xlabel('Informality Rate')
    axes[0, 1].set_ylabel('Final Inflation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PLOT 3: Inflation Volatility
    # Shows how much inflation fluctuates during simulations
    axes[0, 2].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_inflation_volatility'], 
                    'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 2].set_title('Inflation Volatility by Informality Rate')
    axes[0, 2].set_xlabel('Informality Rate')
    axes[0, 2].set_ylabel('Inflation Volatility')
    axes[0, 2].grid(True, alpha=0.3)
    
    # PLOT 4: Credit Access Gap
    # Shows difference in credit access between formal and informal sectors
    axes[1, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_credit_gap'], 
                    'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 0].set_title('Credit Access Gap by Informality Rate')
    axes[1, 0].set_xlabel('Informality Rate')
    axes[1, 0].set_ylabel('Credit Access Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PLOT 5: Formal Sector Production Share
    # Shows what fraction of production comes from formal sector
    axes[1, 1].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_formal_production_ratio'], 
                    'o-', color='green', linewidth=2, markersize=8)
    axes[1, 1].set_title('Formal Sector Production Share by Informality Rate')
    axes[1, 1].set_xlabel('Informality Rate')
    axes[1, 1].set_ylabel('Formal Production Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    # PLOT 6: Policy Effectiveness
    # Shows how far inflation ends up from the target (lower = better)
    axes[1, 2].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_policy_effectiveness'], 
                    'o-', color='brown', linewidth=2, markersize=8)
    axes[1, 2].set_title('Policy Effectiveness by Informality Rate')
    axes[1, 2].set_xlabel('Informality Rate')
    axes[1, 2].set_ylabel('Distance from Inflation Target')
    axes[1, 2].grid(True, alpha=0.3)
    
    # PLOT 7: Economic Stability
    # Combined measure of inflation and policy rate volatility
    axes[2, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_economic_stability'], 
                    'o-', color='navy', linewidth=2, markersize=8)
    axes[2, 0].set_title('Economic Stability by Informality Rate')
    axes[2, 0].set_xlabel('Informality Rate')
    axes[2, 0].set_ylabel('Combined Volatility Measure')
    axes[2, 0].grid(True, alpha=0.3)
    
    # PLOT 8: Convergence Time (only for simulations that converged)
    convergence_data = by_informality_pd[by_informality_pd['avg_convergence_time'].notna()]
    if len(convergence_data) > 0:
        axes[2, 1].plot(convergence_data['informality_rate'], 
                        convergence_data['avg_convergence_time'], 
                        'o-', color='teal', linewidth=2, markersize=8)
        axes[2, 1].set_title('Average Convergence Time by Informality Rate')
        axes[2, 1].set_xlabel('Informality Rate')
        axes[2, 1].set_ylabel('Convergence Time (Steps)')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        # Show message if no convergence data available
        axes[2, 1].text(0.5, 0.5, 'No Convergence Data', ha='center', va='center', 
                        transform=axes[2, 1].transAxes)
    
    # PLOT 9: Production Gap between Formal and Informal Sectors
    axes[2, 2].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_production_gap'], 
                    'o-', color='crimson', linewidth=2, markersize=8)
    axes[2, 2].set_title('Formal-Informal Production Gap by Informality Rate')
    axes[2, 2].set_xlabel('Informality Rate')
    axes[2, 2].set_ylabel('Production Gap (Formal - Informal)')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the main analysis plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_plot_file = os.path.join(output_dir, f'abm_analysis_main_{timestamp}.png')
    plt.savefig(main_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # CREATE CORRELATION HEATMAP
    # Shows relationships between different economic variables
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    corr_plot_file = os.path.join(output_dir, f'correlation_matrix_{timestamp}.png')
    plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # CREATE DETAILED COMPARISON PLOTS
    # More focused analysis comparing different aspects
    print("Creating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Key Metrics by Informality Level', fontsize=14, fontweight='bold')
    
    # Plot 1: Convergence vs Policy Effectiveness
    axes[0, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['convergence_rate'], 
                    'o-', label='Convergence Rate', linewidth=2)
    axes[0, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_policy_effectiveness'], 
                    'o-', label='Policy Ineffectiveness', linewidth=2)
    axes[0, 0].set_title('Convergence vs Policy Effectiveness')
    axes[0, 0].set_xlabel('Informality Rate')
    axes[0, 0].set_ylabel('Rate / Distance from Target')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Inflation Metrics
    axes[0, 1].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_inflation'], 
                    'o-', label='Avg Inflation', linewidth=2)
    axes[0, 1].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_inflation_volatility'], 
                    'o-', label='Inflation Volatility', linewidth=2)
    axes[0, 1].axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Inflation Target')
    axes[0, 1].set_title('Inflation Metrics by Informality')
    axes[0, 1].set_xlabel('Informality Rate')
    axes[0, 1].set_ylabel('Inflation Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Formal Sector Dominance
    axes[1, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_formal_production_ratio'], 
                    'o-', label='Formal Production Share', linewidth=2)
    axes[1, 0].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_formal_credit_ratio'], 
                    'o-', label='Formal Credit Share', linewidth=2)
    axes[1, 0].set_title('Formal Sector Dominance')
    axes[1, 0].set_xlabel('Informality Rate')
    axes[1, 0].set_ylabel('Formal Sector Share')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Economic Gaps and Instability
    axes[1, 1].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_credit_gap'], 
                    'o-', label='Credit Access Gap', linewidth=2)
    axes[1, 1].plot(by_informality_pd['informality_rate'], 
                    by_informality_pd['avg_economic_stability'], 
                    'o-', label='Economic Instability', linewidth=2)
    axes[1, 1].set_title('Gaps and Stability Measures')
    axes[1, 1].set_xlabel('Informality Rate')
    axes[1, 1].set_ylabel('Gap / Instability Measure')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    dist_plot_file = os.path.join(output_dir, f'key_metrics_analysis_{timestamp}.png')
    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved:")
    print(f"  Main analysis: {main_plot_file}")
    print(f"  Correlation matrix: {corr_plot_file}")
    print(f"  Key metrics analysis: {dist_plot_file}")
    
    return main_plot_file, corr_plot_file, dist_plot_file


def create_summary_report(analysis_results, output_dir):
    """
    Create a comprehensive summary report in Markdown format
    
    This function generates a detailed report that includes:
    - Overall statistics summary
    - Key findings and trends
    - Detailed results tables
    - Policy implications
    - Methodology description
    
    Args:
        analysis_results: Dictionary containing all analysis results
        output_dir: Directory where the report will be saved
        
    Returns:
        str: Path to the saved report file
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f'summary_report_{timestamp}.md')
    
    with open(report_file, 'w') as f:
        # Write report header
        f.write("# ABM Parameter Sweep Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # OVERALL STATISTICS SECTION
        f.write("## Overall Statistics\n\n")
        overall = analysis_results['overall']
        f.write(f"- **Total Simulations**: {overall['total_simulations']:,}\n")
        f.write(f"- **Convergence Rate**: {overall['convergence_rate']:.2%}\n")
        f.write(f"- **Average Final Inflation**: {overall['avg_inflation']:.4f} ({overall['avg_inflation']*100:.2f}%)\n")
        f.write(f"- **Inflation Standard Deviation**: {overall['std_inflation']:.4f}\n")
        f.write(f"- **Average Policy Rate**: {overall['avg_policy_rate']:.4f} ({overall['avg_policy_rate']*100:.2f}%)\n")
        f.write(f"- **Average Inflation Volatility**: {overall['avg_inflation_volatility']:.4f}\n")
        f.write(f"- **Average Credit Access Gap**: {overall['avg_credit_gap']:.4f}\n")
        f.write(f"- **Average Formal Production Ratio**: {overall['avg_formal_production_ratio']:.4f}\n\n")
        
        # KEY FINDINGS SECTION
        f.write("## Key Findings\n\n")
        
        # Analyze trends by comparing low vs high informality
        by_informality = analysis_results['by_informality']
        
        # Find representative low and high informality scenarios for comparison
        low_informality = next(r for r in by_informality if r['informality_rate'] <= 0.2)
        high_informality = next(r for r in by_informality if r['informality_rate'] >= 0.7)
        
        f.write("### Impact of Informality on Economic Outcomes\n\n")
        
        # Compare convergence rates
        f.write(f"1. **Convergence**: Models with low informality ({low_informality['informality_rate']:.1f}) show "
                f"{low_informality['convergence_rate']:.1%} convergence rate vs {high_informality['convergence_rate']:.1%} "
                f"for high informality ({high_informality['informality_rate']:.1f}).\n\n")
        
        # Compare inflation control
        f.write(f"2. **Inflation Control**: Low informality scenarios achieve average inflation of "
                f"{low_informality['avg_inflation']:.4f} compared to {high_informality['avg_inflation']:.4f} "
                f"in high informality scenarios.\n\n")
        
        # Compare economic stability
        f.write(f"3. **Economic Stability**: Inflation volatility increases from "
                f"{low_informality['avg_inflation_volatility']:.4f} (low informality) to "
                f"{high_informality['avg_inflation_volatility']:.4f} (high informality).\n\n")
        
        # Compare credit access
        f.write(f"4. **Credit Access**: The credit access gap widens from "
                f"{low_informality['avg_credit_gap']:.4f} to {high_informality['avg_credit_gap']:.4f} "
                f"as informality increases.\n\n")
        
        # DETAILED RESULTS TABLE
        f.write("## Detailed Results by Informality Rate\n\n")
        f.write("| Informality Rate | Convergence Rate | Avg Inflation | Inflation Volatility | Credit Gap | Formal Production % |\n")
        f.write("|------------------|------------------|---------------|---------------------|------------|--------------------|\n")
        
        # Create table rows for each informality rate
        for result in by_informality:
            f.write(f"| {result['informality_rate']:.1f} | "
                   f"{result['convergence_rate']:.1%} | "
                   f"{result['avg_inflation']:.4f} | "
                   f"{result['avg_inflation_volatility']:.4f} | "
                   f"{result['avg_credit_gap']:.4f} | "
                   f"{result['avg_formal_production_ratio']:.1%} |\n")
        
        f.write("\n")
        
        # REGIME ANALYSIS TABLE
        f.write("## Analysis by Informality Regime\n\n")
        f.write("| Regime | Simulations | Convergence Rate | Avg Inflation | Policy Effectiveness |\n")
        f.write("|--------|-------------|------------------|---------------|--------------------|\n")
        
        # Create table for regime comparison (Low/Medium/High)
        for regime in analysis_results['by_regime']:
            f.write(f"| {regime['informality_category']} | "
                   f"{regime['n_simulations']} | "
                   f"{regime['convergence_rate']:.1%} | "
                   f"{regime['avg_inflation']:.4f} | "
                   f"{regime['avg_policy_effectiveness']:.4f} |\n")
        
        f.write("\n")
        
        # POLICY IMPLICATIONS SECTION
        f.write("## Policy Implications\n\n")
        f.write("1. **Monetary Policy Effectiveness**: Higher levels of informality appear to reduce "
                "the effectiveness of monetary policy, as evidenced by lower convergence rates and "
                "higher inflation volatility.\n\n")
        
        f.write("2. **Credit Channel**: The widening credit access gap in high informality scenarios "
                "suggests that informal sectors have limited access to formal credit markets, "
                "potentially reducing monetary policy transmission.\n\n")
        
        f.write("3. **Economic Stability**: Economies with higher informality rates show greater "
                "macroeconomic instability, making inflation targeting more challenging.\n\n")
        
        f.write("4. **Structural Reforms**: The results suggest that reducing informality through "
                "structural reforms could enhance monetary policy effectiveness and economic stability.\n\n")
        
        # METHODOLOGY SECTION
        f.write("## Methodology\n\n")
        f.write("- **Model Type**: Agent-Based Model (ABM) of monetary policy with banking system\n")
        f.write("- **Parameter Sweep**: Informality rates from 0.1 to 0.8 in 8 steps\n")
        f.write("- **Monte Carlo**: 100 random seeds per parameter combination\n")
        f.write(f"- **Total Simulations**: {overall['total_simulations']:,}\n")
        f.write("- **Convergence Criterion**: Inflation stable within [1%, 3%] for 36 consecutive periods\n")
        f.write("- **Analysis Tools**: Apache Spark for scalable data processing\n\n")
    
    print(f"Summary report saved to: {report_file}")
    return report_file


def main():
    """
    Main function that orchestrates the entire analysis process
    
    This function:
    1. Parses command line arguments
    2. Creates and configures Spark session
    3. Loads and prepares the data
    4. Runs comprehensive analysis
    5. Creates visualizations
    6. Generates summary report
    7. Cleans up Spark resources
    """
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze ABM Parameter Sweep Results with Spark')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV results file from parameter sweep')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Output directory for analysis results (default: analysis_results)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Spark session
    print("Initializing Spark session...")
    spark = create_spark_session()
    
    try:
        # Load and prepare data for analysis
        print("Loading and preparing data...")
        df = load_and_prepare_data(spark, args.csv_file)
        
        print(f"Loaded {df.count()} simulation results")
        print("Data schema:")
        df.printSchema()  # Shows the structure of our data
        
        # Perform comprehensive statistical analysis
        analysis_results, by_informality_pd, correlation_matrix = comprehensive_analysis(df, args.output_dir)
        
        # Create all visualizations
        plots = create_visualizations(by_informality_pd, correlation_matrix, args.output_dir)
        
        # Generate summary report
        report_file = create_summary_report(analysis_results, args.output_dir)
        
        # Print completion message with file locations
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Summary report: {report_file}")
        
    finally:
        # Always stop Spark session to free up resources
        # This is important in cluster environments
        spark.stop()


# Entry point - only run if script is executed directly
if __name__ == "__main__":
    main()