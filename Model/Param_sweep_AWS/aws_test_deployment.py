#!/usr/bin/env python3
"""
Test AWS deployment with minimal parameters
"""

import json
import sys
import os
from datetime import datetime

def create_test_config():
    """Create test configuration for AWS deployment"""
    
    # Check if we have an existing AWS config
    if os.path.exists('aws_sweep_config.json'):
        with open('aws_sweep_config.json', 'r') as f:
            base_config = json.load(f)
    else:
        print("❌ AWS configuration not found. Please run aws_setup_script.py first.")
        return None
    
    # Create minimal test configuration
    test_config = base_config.copy()
    test_config.update({
        "informality_rates": [0.1, 0.3, 0.5],  # Only 3 rates
        "seeds": list(range(2)),                # Only 2 seeds
        "model_params": {
            "n_firms": 30,                      # Smaller model
            "n_consumers": 100,
            "n_banks": 3,
            "max_steps": 50,                    # Fewer steps
            "inflation_target": 0.02,
            "initial_policy_rate": 0.03,
            "current_inflation": 0.02
        },
        "aws_config": {
            **base_config["aws_config"],
            "num_instances": 1,                 # Only 1 instance for test
            "estimated_runtime_hours": 0.5,    # 30 minutes max
            "auto_confirm": True                # Skip confirmation
        }
    })
    
    # Save test config
    test_config_file = 'aws_test_config.json'
    with open(test_config_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"✅ Test configuration created: {test_config_file}")
    print(f"Test will run: {len(test_config['informality_rates'])} × {len(test_config['seeds'])} = {len(test_config['informality_rates']) * len(test_config['seeds'])} simulations")
    print(f"Estimated cost: ~$0.17 (30 minutes on 1 instance)")
    
    return test_config_file

def run_aws_test():
    """Run AWS test deployment"""
    
    print("ABM Parameter Sweep - AWS Test Deployment")
    print("="*50)
    
    # Create test configuration
    test_config_file = create_test_config()
    if not test_config_file:
        return False
    
    # Import and run main runner
    try:
        from main_runner import load_config, validate_config, run_aws_sweep
        
        # Load test config
        config = load_config(test_config_file)
        
        # Validate configuration
        if not validate_config(config):
            print("❌ Test configuration validation failed")
            return False
        
        print("\n📋 Test Configuration:")
        print(f"- Mode: {config['mode']}")
        print(f"- Simulations: {len(config['informality_rates']) * len(config['seeds'])}")
        print(f"- AWS Region: {config['aws_config']['region_name']}")
        print(f"- Instance Type: {config['aws_config']['instance_type']}")
        print(f"- S3 Bucket: {config['aws_config']['s3_bucket']}")
        
        # Confirm test run
        if not config['aws_config'].get('auto_confirm', False):
            confirm = input("\n🚀 Run AWS test deployment? (y/n): ").lower().strip()
            if confirm != 'y':
                print("Test cancelled.")
                return False
        
        print("\n🚀 Starting AWS test deployment...")
        
        # Run AWS sweep
        results = run_aws_sweep(config)
        
        if results:
            print("✅ AWS test deployment completed successfully!")
            print(f"Result files: {results}")
            return True
        else:
            print("❌ AWS test deployment failed")
            return False
            
    except Exception as e:
        print(f"❌ Test deployment error: {str(e)}")
        return False

def main():
    """Main test function"""
    
    # Check prerequisites
    try:
        import boto3
        print("✅ boto3 available")
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        return
    
    # Check AWS credentials
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials valid. Account: {identity.get('Account')}")
    except Exception as e:
        print(f"❌ AWS credentials issue: {str(e)}")
        print("Please run: aws configure")
        return
    
    # Run test
    success = run_aws_test()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nNext steps:")
        print("1. Check your S3 bucket for results")
        print("2. If satisfied, run full deployment:")
        print("   python main_runner.py --config aws_sweep_config.json")
    else:
        print("\n❌ Test failed. Please check error messages and try again.")

if __name__ == "__main__":
    main()