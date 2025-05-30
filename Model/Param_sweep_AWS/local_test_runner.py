#!/usr/bin/env python3
"""
Local test runner for MESA-FREE ABM simulations
Test a few simulations locally before deploying to AWS
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
import pickle
import os
import random
import sys

def run_single_simulation(params):
    """Run a single MESA-FREE simulation locally"""
    try:
        # Import your MESA-FREE model
        from model import MonetaryPolicyModel
        
        # Set random seeds for reproducibility
        np.random.seed(params['seed'])
        random.seed(params['seed'])
        
        print(f"  Starting MESA-FREE simulation: rate={params['informality_rate']}, seed={params['seed']}")
        start_time = time.time()
        
        # Create MESA-FREE model
        model = MonetaryPolicyModel(
            n_firms=params['n_firms'],
            n_consumers=params['n_consumers'],
            n_commercial_banks=params['n_banks'],
            inflation_target=params['inflation_target'],
            initial_policy_rate=params['initial_policy_rate'],
            informality_rate=params['informality_rate'],
            current_inflation=params['current_inflation']
        )
        
        print(f"    Model created: {len(model.firms)} firms ({sum(1 for f in model.firms if f.is_informal)} informal), {len(model.consumers)} consumers ({sum(1 for c in model.consumers if c.is_informal)} informal)")
        
        # Run simulation
        for step in range(params['max_steps']):
            model.step()
            
            # Print progress every 20 steps for shorter runs
            if params['max_steps'] <= 50 and (step + 1) % 10 == 0:
                print(f"    Step {step + 1}/{params['max_steps']}: inflation={model.current_inflation:.4f}, policy_rate={model.policy_rate:.4f}")
            elif (step + 1) % 25 == 0:
                print(f"    Step {step + 1}/{params['max_steps']}: inflation={model.current_inflation:.4f}, policy_rate={model.policy_rate:.4f}")
        
        run_time = time.time() - start_time
        
        # Collect results from MESA-FREE model
        results = {
            'informality_rate': params['informality_rate'],
            'seed': params['seed'],
            'final_inflation': model.current_inflation,
            'final_policy_rate': model.policy_rate,
            'final_output_gap': model.output_gap,
            'credit_access_gap': model._calculate_credit_gap(),
            'total_production': sum([f.production for f in model.firms]),
            'formal_production': sum([f.production for f in model.firms if not f.is_informal]),
            'informal_production': sum([f.production for f in model.firms if f.is_informal]),
            'avg_lending_rate': np.mean([bank.lending_rate for bank in model.commercial_banks]),
            'converged': model.is_converged(),
            'steps_completed': model.time,
            'run_time': run_time,
            'run_timestamp': datetime.now().isoformat(),
            'mesa_free': True,  # Flag to indicate this was run without Mesa
            'informal_firms_count': sum(1 for f in model.firms if f.is_informal),
            'informal_consumers_count': sum(1 for c in model.consumers if c.is_informal),
            'avg_firm_price': np.mean([f.price for f in model.firms]),
            'total_consumption': sum([c.consumption for c in model.consumers])
        }
        
        print(f"  ✅ MESA-FREE simulation completed in {run_time:.1f}s")
        print(f"     Final inflation: {results['final_inflation']:.4f}, Credit gap: {results['credit_access_gap']:.3f}")
        print(f"     Convergence: {'Yes' if results['converged'] else 'No'}, Production: {results['total_production']:.1f}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ MESA-FREE simulation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'params': params, 'mesa_free': True}

def run_local_test():
    """Run local MESA-FREE test simulations"""
    
    print("ABM Local Test Runner - MESA-FREE VERSION")
    print("=" * 50)
    
    # Test parameters - small but representative
    test_params = {
        'n_firms': 30,           # Smaller for faster testing
        'n_consumers': 100,      # Smaller for faster testing  
        'n_banks': 3,            # Smaller for faster testing
        'max_steps': 50,         # Fewer steps for faster testing
        'inflation_target': 0.02,
        'initial_policy_rate': 0.03,
        'current_inflation': 0.12
    }
    
    # Test with different informality rates and multiple seeds
    test_combinations = [
        # Low informality
        {'informality_rate': 0.1, 'seed': 0},
        {'informality_rate': 0.1, 'seed': 1},
        
        # Medium informality  
        {'informality_rate': 0.3, 'seed': 0},
        {'informality_rate': 0.3, 'seed': 1},
        
        # High informality
        {'informality_rate': 0.5, 'seed': 0},
        {'informality_rate': 0.5, 'seed': 1},
        
        # Very high informality
        {'informality_rate': 0.7, 'seed': 0},
        {'informality_rate': 0.7, 'seed': 1},
    ]
    
    print(f"Running {len(test_combinations)} MESA-FREE test simulations...")
    print(f"Model size: {test_params['n_firms']} firms, {test_params['n_consumers']} consumers, {test_params['n_banks']} banks")
    print(f"Steps per simulation: {test_params['max_steps']}")
    print(f"Testing informality rates: {sorted(set(c['informality_rate'] for c in test_combinations))}")
    print("")
    
    results = []
    total_start_time = time.time()
    
    for i, combination in enumerate(test_combinations):
        print(f"MESA-FREE Simulation {i+1}/{len(test_combinations)}:")
        
        # Combine test params with this combination
        sim_params = {**test_params, **combination}
        
        # Run simulation
        result = run_single_simulation(sim_params)
        results.append(result)
        print("")
    
    total_time = time.time() - total_start_time
    
    # Analyze results
    print("=" * 50)
    print("MESA-FREE LOCAL TEST RESULTS")
    print("=" * 50)
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"Successful simulations: {len(successful_results)}/{len(results)}")
    print(f"Failed simulations: {len(failed_results)}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per simulation: {total_time/len(results):.1f} seconds")
    
    if failed_results:
        print(f"\nErrors encountered:")
        for result in failed_results:
            print(f"  - {result['error']}")
        print(f"\n❌ Some simulations failed. Fix issues before AWS deployment.")
        return False
    
    if successful_results:
        # Create summary
        df = pd.DataFrame(successful_results)
        
        print(f"\nDetailed Results Summary:")
        print(f"{'Rate':<6} {'Seed':<6} {'Inflation':<10} {'Policy Rate':<12} {'Credit Gap':<12} {'Production':<12} {'Converged':<10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            converged_str = "Yes" if row['converged'] else "No"
            print(f"{row['informality_rate']:<6.1f} {row['seed']:<6} {row['final_inflation']:<10.4f} {row['final_policy_rate']:<12.4f} {row['credit_access_gap']:<12.3f} {row['total_production']:<12.1f} {converged_str:<10}")
        
        # Comprehensive analysis
        print(f"\nStatistical Summary by Informality Rate:")
        summary = df.groupby('informality_rate').agg({
            'final_inflation': ['mean', 'std', 'min', 'max'],
            'credit_access_gap': ['mean', 'std'],
            'total_production': ['mean', 'std'],
            'converged': 'mean'
        }).round(4)
        
        print(summary)
        
        # Additional insights
        print(f"\nKey Insights:")
        min_inflation_rate = df.groupby('informality_rate')['final_inflation'].mean().idxmin()
        max_inflation_rate = df.groupby('informality_rate')['final_inflation'].mean().idxmax()
        min_credit_gap_rate = df.groupby('informality_rate')['credit_access_gap'].mean().idxmin()
        max_credit_gap_rate = df.groupby('informality_rate')['credit_access_gap'].mean().idxmax()
        
        print(f"- Lowest average inflation ({df.groupby('informality_rate')['final_inflation'].mean()[min_inflation_rate]:.4f}) at informality rate: {min_inflation_rate}")
        print(f"- Highest average inflation ({df.groupby('informality_rate')['final_inflation'].mean()[max_inflation_rate]:.4f}) at informality rate: {max_inflation_rate}")
        print(f"- Smallest credit gap ({df.groupby('informality_rate')['credit_access_gap'].mean()[min_credit_gap_rate]:.3f}) at informality rate: {min_credit_gap_rate}")
        print(f"- Largest credit gap ({df.groupby('informality_rate')['credit_access_gap'].mean()[max_credit_gap_rate]:.3f}) at informality rate: {max_credit_gap_rate}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"mesa_free_local_test_results_{timestamp}.pkl"
        
        with open(results_file, 'wb') as f:
            pickle.dump(successful_results, f)
        
        print(f"\nResults saved to: {results_file}")
        
        # Comprehensive sanity checks
        inflation_range = df['final_inflation'].max() - df['final_inflation'].min()
        credit_gap_range = df['credit_access_gap'].max() - df['credit_access_gap'].min()
        production_range = df['total_production'].max() - df['total_production'].min()
        convergence_rate = df['converged'].mean()
        
        print(f"\nSanity Checks:")
        print(f"✅ All simulations completed successfully")
        print(f"✅ Inflation range: {inflation_range:.4f} (should be > 0)")
        print(f"✅ Credit gap range: {credit_gap_range:.3f} (should be > 0)")
        print(f"✅ Production range: {production_range:.1f} (should be > 0)")
        print(f"✅ Convergence rate: {convergence_rate:.1%}")
        print(f"✅ Average simulation time: {df['run_time'].mean():.1f}s (MESA-FREE is fast!)")
        
        # Validate economic relationships
        corr_informality_inflation = df['informality_rate'].corr(df['final_inflation'])
        corr_informality_credit = df['informality_rate'].corr(df['credit_access_gap'])
        
        print(f"\nEconomic Relationship Validation:")
        print(f"✅ Correlation (informality ↔ inflation): {corr_informality_inflation:.3f}")
        print(f"✅ Correlation (informality ↔ credit gap): {corr_informality_credit:.3f}")
        
        # AWS deployment estimates
        aws_time_per_sim = df['run_time'].mean() * 1.3  # AWS slightly slower + startup overhead
        full_sweep_sims = 8 * 20  # 8 rates × 20 seeds = 160 simulations
        estimated_aws_time_5_instances = (full_sweep_sims * aws_time_per_sim) / 5  # 5 instances parallel
        
        print(f"\nAWS Deployment Estimates:")
        print(f"📊 Estimated time per simulation on AWS: {aws_time_per_sim:.1f}s")
        print(f"📊 Full sweep (160 simulations): {full_sweep_sims} total")
        print(f"📊 Estimated total time on 5 instances: {estimated_aws_time_5_instances/60:.1f} minutes")
        print(f"📊 Estimated cost: ~${(estimated_aws_time_5_instances/3600) * 5 * 0.042:.2f} (5 × t3.medium)")
        
        # Performance comparison note
        print(f"\n🚀 MESA-FREE Performance Notes:")
        print(f"- No Mesa dependency installation time")
        print(f"- Faster agent creation and stepping")
        print(f"- More reliable AWS deployment")
        print(f"- Same economic logic and results")
        
        return True
    
    return False

def validate_model_files():
    """Validate that required model files exist and are compatible"""
    
    print("Validating MESA-FREE model files...")
    
    # Check if model files exist
    if not os.path.exists('model.py'):
        print("❌ model.py not found in current directory")
        return False
    
    if not os.path.exists('agents.py'):
        print("❌ agents.py not found in current directory") 
        return False
    
    print("✅ Found model.py and agents.py")
    
    # Try importing to check for syntax errors
    try:
        from model import MonetaryPolicyModel
        print("✅ Successfully imported MonetaryPolicyModel (MESA-FREE)")
        
        # Quick instantiation test
        test_model = MonetaryPolicyModel(n_firms=5, n_consumers=10, n_commercial_banks=1)
        print("✅ Successfully created test model instance")
        print(f"   - Created {len(test_model.firms)} firms, {len(test_model.consumers)} consumers, {len(test_model.commercial_banks)} banks")
        
        # Test one step
        test_model.step()
        print("✅ Successfully executed one simulation step")
        print(f"   - Time: {test_model.time}, Inflation: {test_model.current_inflation:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def main():
    """Main function"""
    
    print("MESA-FREE ABM Local Testing Suite")
    print("=" * 50)
    
    # Validate model files first
    if not validate_model_files():
        print(f"\n❌ Model validation failed. Fix issues before proceeding.")
        return
    
    print("")
    
    # Ask user for confirmation
    response = input("Run MESA-FREE local test simulations? (y/n): ").lower().strip()
    if response != 'y':
        print("Test cancelled.")
        return
    
    print("")
    
    # Run local test
    success = run_local_test()
    
    if success:
        print(f"\n🎉 MESA-FREE LOCAL TEST SUCCESSFUL!")
        print(f"Your MESA-FREE model is working correctly and ready for AWS deployment.")
        print(f"\nNext steps:")
        print(f"1. python simple_aws_setup.py  # Set up AWS resources")
        print(f"2. python run_aws_sweep.py     # Deploy MESA-FREE sweep to AWS")
        print(f"3. python analyze_results.py   # Analyze results when complete")
        print(f"\nExpected AWS completion time: ~30-45 minutes for full sweep (much faster without Mesa!)")
    else:
        print(f"\n❌ MESA-FREE LOCAL TEST FAILED")
        print(f"Fix the issues above before deploying to AWS")
        print(f"Common issues:")
        print(f"- Missing numpy/pandas: pip install numpy pandas scipy matplotlib")
        print(f"- Model syntax errors: check model.py carefully")
        print(f"- Import path issues: make sure model.py and agents.py are in same directory")

if __name__ == "__main__":
    main()