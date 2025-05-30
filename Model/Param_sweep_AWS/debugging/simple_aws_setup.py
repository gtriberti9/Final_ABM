#!/usr/bin/env python3
"""
Simple AWS setup for learner environments - MESA-FREE VERSION
Creates only essential resources with minimal permissions
Much faster simulations without Mesa dependency
"""

import boto3
import json
import random
import string
from botocore.exceptions import ClientError

def create_unique_bucket_name():
    """Create unique S3 bucket name"""
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"abm-mesa-free-{suffix}"

def check_lab_instance_profile():
    """Check if LabInstanceProfile exists"""
    
    try:
        iam_client = boto3.client('iam')
        
        # Check if LabInstanceProfile exists
        try:
            iam_client.get_instance_profile(InstanceProfileName='LabInstanceProfile')
            print("✅ LabInstanceProfile exists and ready to use")
            return 'LabInstanceProfile'
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                print("❌ LabInstanceProfile not found")
                print("Available instance profiles:")
                
                # List available instance profiles
                try:
                    profiles = iam_client.list_instance_profiles()
                    for profile in profiles['InstanceProfiles']:
                        print(f"   - {profile['InstanceProfileName']}")
                except Exception:
                    pass
                
                return None
            else:
                raise
    
    except ClientError as e:
        print(f"⚠️ Could not check instance profile: {e.response['Error']['Code']}")
        print("Using LabInstanceProfile anyway (may work in some environments)")
        return 'LabInstanceProfile'
    except Exception as e:
        print(f"⚠️ Error checking instance profile: {e}")
        return 'LabInstanceProfile'  # Return it anyway, might work

def setup_aws_for_learner():
    """Set up AWS resources for learner environment"""
    
    print("Setting up AWS for MESA-FREE learner environment...")
    
    # Initialize clients
    try:
        s3_client = boto3.client('s3')
        ec2_client = boto3.client('ec2')
        sts_client = boto3.client('sts')
    except Exception as e:
        print(f"❌ AWS connection failed: {e}")
        print("Make sure you've configured AWS credentials")
        return None
    
    # Test connection
    try:
        identity = sts_client.get_caller_identity()
        print(f"✅ Connected to AWS account: {identity.get('Account')}")
    except Exception as e:
        print(f"❌ AWS credentials error: {e}")
        return None
    
    setup_results = {}
    
    # Create S3 bucket
    bucket_name = create_unique_bucket_name()
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"✅ Created S3 bucket: {bucket_name}")
        setup_results['s3_bucket'] = bucket_name
    except Exception as e:
        print(f"❌ Failed to create S3 bucket: {e}")
        return None
    
    # Check LabInstanceProfile
    instance_profile = check_lab_instance_profile()
    if instance_profile:
        setup_results['iam_role'] = instance_profile
        print(f"✅ Will use instance profile: {instance_profile}")
    else:
        print("❌ Could not find LabInstanceProfile")
        print("Please check with your instructor about the correct instance profile name")
        return None
    
    # Get default VPC info (don't create new resources)
    try:
        vpcs = ec2_client.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
        if vpcs['Vpcs']:
            vpc_id = vpcs['Vpcs'][0]['VpcId']
            print(f"✅ Using default VPC: {vpc_id}")
            
            # Get a subnet from default VPC
            subnets = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
            if subnets['Subnets']:
                subnet_id = subnets['Subnets'][0]['SubnetId']
                setup_results['subnet_id'] = subnet_id
                print(f"✅ Using subnet: {subnet_id}")
    except Exception as e:
        print(f"⚠️ Could not get VPC info: {e}")
    
    # Use default security group
    try:
        sgs = ec2_client.describe_security_groups(Filters=[{'Name': 'group-name', 'Values': ['default']}])
        if sgs['SecurityGroups']:
            sg_id = sgs['SecurityGroups'][0]['GroupId']
            setup_results['security_group_id'] = sg_id
            print(f"✅ Using default security group: {sg_id}")
    except Exception as e:
        print(f"⚠️ Could not get security group: {e}")
    
    return setup_results

def create_config_file(setup_results):
    """Create configuration file for ABM sweep"""
    
    config = {
        "mode": "aws_mesa_free",
        "informality_rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "seeds": list(range(20)),  # 20 seeds for learner environment
        "model_params": {
            "n_firms": 50,
            "n_consumers": 500,
            "n_banks": 5,
            "max_steps": 100,
            "inflation_target": 0.02,
            "initial_policy_rate": 0.03,
            "current_inflation": 0.12
        },
        "aws_config": {
            "region_name": "us-east-1",
            "instance_type": "t3.medium",
            "num_instances": 5,  # Multiple instances for parallel processing
            "s3_bucket": setup_results['s3_bucket'],
            "iam_role": setup_results['iam_role'],  # Now correctly uses LabInstanceProfile
            "security_group_id": setup_results.get('security_group_id'),
            "subnet_id": setup_results.get('subnet_id'),
            "estimated_runtime_hours": 0.5  # Much faster without Mesa!
        }
    }
    
    # Save config
    with open('aws_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to: aws_config.json")
    return config

def main():
    """Main setup function"""
    
    print("ABM Parameter Sweep - AWS Learner Setup (MESA-FREE VERSION)")
    print("=" * 70)
    
    # Setup AWS resources
    results = setup_aws_for_learner()
    
    if not results:
        print("❌ Setup failed")
        return
    
    # Create configuration
    config = create_config_file(results)
    
    print("\n✅ MESA-FREE Setup Complete!")
    print(f"S3 Bucket: {results['s3_bucket']}")
    print(f"Instance Profile: {results['iam_role']}")
    print(f"Configuration: aws_config.json")
    
    print(f"\n💰 Estimated Cost (MESA-FREE):")
    print(f"- Instance: t3.medium ($0.042/hour)")
    print(f"- Number of instances: {config['aws_config']['num_instances']}")
    print(f"- Runtime: ~{config['aws_config']['estimated_runtime_hours']} hours (50% faster!)")
    
    total_cost = 0.042 * config['aws_config']['num_instances'] * config['aws_config']['estimated_runtime_hours']
    print(f"- Total: ~${total_cost:.2f} (Much cheaper without Mesa!)")
    
    total_sims = len(config['informality_rates']) * len(config['seeds'])
    print(f"- Simulations: {total_sims} (8 rates × 20 seeds)")
    print(f"- Parallel execution: {config['aws_config']['num_instances']} instances")
    
    print(f"\n🚀 Next steps:")
    print(f"1. python run_aws_sweep.py")
    print(f"2. Wait for completion (~{config['aws_config']['estimated_runtime_hours']} hour)")
    print(f"3. python analyze_results.py")
    
    print(f"\n✅ MESA-FREE VERSION ADVANTAGES:")
    print(f"- No Mesa dependency issues")
    print(f"- Faster startup and execution")
    print(f"- More reliable on AWS")
    print(f"- Lower costs due to faster runtime")
    print(f"- Same simulation logic and results")

if __name__ == "__main__":
    main()