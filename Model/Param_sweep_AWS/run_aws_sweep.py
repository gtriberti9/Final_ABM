#!/usr/bin/env python3
"""
Multi-instance AWS parameter sweep runner - MESA-FREE VERSION
Much faster and more reliable without Mesa dependency
"""

import json
import boto3
import time
import zipfile
import os
import math
from datetime import datetime
from botocore.exceptions import ClientError

def load_config():
    """Load configuration file"""
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Configuration file not found. Run simple_aws_setup.py first.")
        return None

def create_deployment_package():
    """Create deployment package with necessary files"""
    
    package_name = "abm_deployment.zip"
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add your model files (make sure these exist)
        required_files = ['model.py', 'agents.py']
        
        for file in required_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"SUCCESS Added {file} to package")
            else:
                print(f"ERROR Missing required file: {file}")
                return None
        
        # Create FAST startup script with progress logging (NO MESA, NO UNICODE)
        runner_script = '''#!/usr/bin/env python3
import sys
import os
import time
sys.path.append('/home/ec2-user')

print("=== STARTUP SCRIPT BEGIN (MESA-FREE) ===")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Create progress log function
def log_progress(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

log_progress("Installing packages (MINIMAL DEPENDENCIES)...")

# Install only essential packages
packages = ["numpy", "pandas", "matplotlib", "scipy"]
for package in packages:
    log_progress(f"Installing {package}...")
    result = os.system(f"pip3 install --user {package} --quiet")
    if result == 0:
        log_progress(f"SUCCESS {package} installed successfully")
    else:
        log_progress(f"ERROR {package} installation failed")

log_progress("Package installation complete")

# Test imports (NO MESA IMPORT)
log_progress("Testing imports...")
try:
    import numpy as np
    log_progress("SUCCESS numpy imported")
    import pandas as pd
    log_progress("SUCCESS pandas imported")
    log_progress("All imports successful - MESA-FREE MODEL READY")
except Exception as e:
    log_progress(f"ERROR Import error: {e}")

log_progress("Starting simulation script...")

# Import and run parameter sweep
exec(open("run_simulations.py").read())
'''
        
        with open('startup.py', 'w') as f:
            f.write(runner_script)
        zipf.write('startup.py')
        
        # Create MESA-FREE simulation runner 
        sim_script_template = '''
import sys
import os
import json
import pickle
import time
from datetime import datetime
sys.path.append('/home/ec2-user')

def log_progress(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

log_progress("=== MESA-FREE SIMULATION SCRIPT BEGIN ===")

try:
    # Import our MESA-FREE model
    from model import MonetaryPolicyModel
    import numpy as np
    log_progress("Successfully imported MESA-FREE model and numpy")
except Exception as e:
    log_progress(f"Import error: {e}")
    sys.exit(1)

def run_single_simulation(params):
    """Run a single MESA-FREE simulation"""
    try:
        log_progress(f"Starting MESA-FREE simulation: rate={params['informality_rate']}, seed={params['seed']}")
        
        # Set random seeds for reproducibility
        np.random.seed(params['seed'])
        import random
        random.seed(params['seed'])
        
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
        
        log_progress(f"MESA-FREE model created, running {params['max_steps']} steps...")
        
        # Run simulation with progress updates
        for step in range(params['max_steps']):
            model.step()
            # Log progress every 10 steps for small simulations
            if params['max_steps'] <= 20 and (step + 1) % 5 == 0:
                log_progress(f"  Step {step + 1}/{params['max_steps']}")
            elif (step + 1) % 25 == 0:
                log_progress(f"  Step {step + 1}/{params['max_steps']}")
        
        # Collect results from MESA-FREE model
        results = {
            'informality_rate': params['informality_rate'],
            'seed': params['seed'],
            'instance_id': params.get('instance_id', 0),
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
            'run_timestamp': datetime.now().isoformat(),
            'mesa_free': True  # Flag to indicate this was run without Mesa
        }
        
        log_progress(f"SUCCESS MESA-FREE simulation completed: inflation={results['final_inflation']:.4f}")
        return results
        
    except Exception as e:
        log_progress(f"ERROR MESA-FREE simulation error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'params': params, 'mesa_free': True}

def main():
    """Main simulation runner"""
    
    log_progress("=== MESA-FREE MAIN FUNCTION START ===")
    
    # Get instance metadata (simplified, no requests dependency)
    try:
        # Try to get instance ID without requests library
        import urllib.request
        response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
        instance_id = response.read().decode('utf-8')
        log_progress(f"Running on instance: {instance_id}")
    except Exception as e:
        instance_id = "unknown"
        log_progress(f"Could not get instance ID: {e}")
    
    # Configuration will be replaced by deployment script
    config = CONFIG_PLACEHOLDER
    
    log_progress(f"Configuration loaded: {len(config['informality_rates'])} rates, {len(config['seeds'])} seeds")
    
    # Calculate which simulations this instance should run
    total_sims = len(config['informality_rates']) * len(config['seeds'])
    num_instances = config['aws_config']['num_instances']
    
    # Get instance index from environment variable (set by deployment script)
    instance_index = int(os.environ.get('INSTANCE_INDEX', '0'))
    
    # Calculate simulation range for this instance
    sims_per_instance = math.ceil(total_sims / num_instances)
    start_sim = instance_index * sims_per_instance
    end_sim = min(start_sim + sims_per_instance, total_sims)
    
    log_progress(f"Instance {instance_index} will run MESA-FREE simulations {start_sim} to {end_sim-1}")
    
    # Create flat list of all parameter combinations
    all_combinations = []
    for informality_rate in config['informality_rates']:
        for seed in config['seeds']:
            all_combinations.append({
                **config['model_params'],
                'informality_rate': informality_rate,
                'seed': seed,
                'instance_id': instance_index
            })
    
    # Get this instance's subset
    my_combinations = all_combinations[start_sim:end_sim]
    
    log_progress(f"Starting MESA-FREE parameter sweep on instance {instance_index}")
    log_progress(f"Running {len(my_combinations)} simulations out of {total_sims} total")
    
    results = []
    
    # Run assigned simulations
    for i, params in enumerate(my_combinations):
        log_progress(f"=== MESA-FREE SIMULATION {i+1}/{len(my_combinations)} ===")
        
        start_time = time.time()
        result = run_single_simulation(params)
        run_time = time.time() - start_time
        
        results.append(result)
        log_progress(f"MESA-FREE simulation {i+1} completed in {run_time:.1f}s")
    
    # Save results with instance identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"abm_results_mesa_free_instance_{instance_index}_{timestamp}.pkl"
    
    log_progress(f"Saving MESA-FREE results to {results_file}")
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    log_progress(f"MESA-FREE results saved locally")
    
    # Upload to S3
    log_progress("Uploading MESA-FREE results to S3...")
    try:
        import boto3
        s3_client = boto3.client('s3')
        bucket_name = config['aws_config']['s3_bucket']
        
        s3_client.upload_file(results_file, bucket_name, f"results/{results_file}")
        log_progress(f"SUCCESS MESA-FREE results uploaded to S3: s3://{bucket_name}/results/{results_file}")
        
        # Create completion marker for this instance
        completion_marker = f"COMPLETED_MESA_FREE_INSTANCE_{instance_index}_{timestamp}.txt"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"results/{completion_marker}",
            Body=f"Instance {instance_index} completed {len(results)} MESA-FREE simulations at {datetime.now()}"
        )
        log_progress(f"SUCCESS MESA-FREE completion marker uploaded: {completion_marker}")
        
    except Exception as e:
        log_progress(f"ERROR Failed to upload to S3: {e}")
        log_progress("MESA-FREE results are saved locally on the instance")
    
    log_progress(f"SUCCESS Instance {instance_index} MESA-FREE parameter sweep completed!")
    log_progress("=== MESA-FREE SIMULATION SCRIPT END ===")

if __name__ == "__main__":
    import math  # Make sure math is imported
    main()
'''
        
        with open('run_simulations.py', 'w') as f:
            f.write(sim_script_template)
        zipf.write('run_simulations.py')
        
        os.remove('startup.py')
        os.remove('run_simulations.py')
    
    print(f"SUCCESS Created MESA-FREE deployment package: {package_name}")
    return package_name

def upload_package_to_s3(package_name, bucket_name):
    """Upload deployment package to S3"""
    
    try:
        s3_client = boto3.client('s3')
        s3_key = f"deployments/{package_name}"
        
        s3_client.upload_file(package_name, bucket_name, s3_key)
        print(f"SUCCESS Package uploaded to S3: s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
        
    except Exception as e:
        print(f"ERROR Failed to upload package: {e}")
        return None

def create_user_data_script(config, s3_package_url, instance_index):
    """Create user data script for EC2 instance"""
    
    # Replace the CONFIG_PLACEHOLDER with actual config
    config_json = json.dumps(config)
    
    user_data = f'''#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "=== USER DATA SCRIPT START (MESA-FREE) ==="
date

yum update -y
yum install -y python3 python3-pip aws-cli unzip

echo "=== Setting up MESA-FREE environment ==="
# Set instance index environment variable
export INSTANCE_INDEX={instance_index}
echo "export INSTANCE_INDEX={instance_index}" >> /home/ec2-user/.bashrc

# Download deployment package
echo "=== Downloading MESA-FREE deployment package ==="
cd /home/ec2-user
aws s3 cp {s3_package_url} deployment.zip
unzip deployment.zip
chown -R ec2-user:ec2-user /home/ec2-user

echo "=== Configuring MESA-FREE simulation script ==="
# Replace config placeholder in the simulation script
python3 -c "
import json
config = {config_json}
with open('run_simulations.py', 'r') as f:
    content = f.read()
content = content.replace('CONFIG_PLACEHOLDER', repr(config))
with open('run_simulations.py', 'w') as f:
    f.write(content)
print('MESA-FREE configuration injected successfully')
"

echo "=== Starting MESA-FREE simulation ==="
# Run as ec2-user with output logging
sudo -u ec2-user -E python3 startup.py

# Signal completion
echo "MESA_FREE_SWEEP_COMPLETE_INSTANCE_{instance_index}" > /tmp/sweep_status.txt
echo "=== USER DATA SCRIPT END (MESA-FREE) ==="
date
'''
    
    return user_data

def launch_ec2_instances(config):
    """Launch multiple EC2 instances for parameter sweep"""
    
    try:
        ec2_client = boto3.client('ec2')
        num_instances = config['aws_config']['num_instances']
        
        # Upload deployment package first
        package_name = create_deployment_package()
        if not package_name:
            return []
        
        s3_url = upload_package_to_s3(package_name, config['aws_config']['s3_bucket'])
        if not s3_url:
            return []
        
        launched_instances = []
        
        # Launch instances one by one with different instance indices
        for instance_index in range(num_instances):
            print(f"Launching MESA-FREE instance {instance_index + 1}/{num_instances}...")
            
            # Create user data script for this specific instance
            user_data = create_user_data_script(config, s3_url, instance_index)
            
            # Instance configuration
            instance_config = {
                'ImageId': 'ami-0c02fb55956c7d316',  # Amazon Linux 2
                'InstanceType': config['aws_config']['instance_type'],
                'MinCount': 1,
                'MaxCount': 1,
                'UserData': user_data,
                'IamInstanceProfile': {'Name': config['aws_config']['iam_role']},
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'ABM-MESA-FREE-Instance-{instance_index}'},
                        {'Key': 'Project', 'Value': 'ABM-MESA-FREE-Multi-Instance'},
                        {'Key': 'InstanceIndex', 'Value': str(instance_index)}
                    ]
                }]
            }
            
            # Add networking if available
            if config['aws_config'].get('security_group_id'):
                instance_config['SecurityGroupIds'] = [config['aws_config']['security_group_id']]
            
            if config['aws_config'].get('subnet_id'):
                instance_config['SubnetId'] = config['aws_config']['subnet_id']
            
            # Launch instance
            response = ec2_client.run_instances(**instance_config)
            instance_id = response['Instances'][0]['InstanceId']
            launched_instances.append(instance_id)
            
            print(f"SUCCESS Launched MESA-FREE instance {instance_index}: {instance_id}")
            
            # Brief pause between launches
            time.sleep(2)
        
        # Cleanup local package
        if os.path.exists(package_name):
            os.remove(package_name)
        
        return launched_instances
        
    except Exception as e:
        print(f"ERROR Failed to launch instances: {e}")
        return []

def monitor_instances(instance_ids, s3_bucket, num_instances):
    """Monitor instances and check for completion"""
    
    ec2_client = boto3.client('ec2')
    s3_client = boto3.client('s3')
    
    print(f"Monitoring {len(instance_ids)} MESA-FREE instances...")
    print(f"Expected completion time: ~{config['aws_config']['estimated_runtime_hours']} hours")
    
    start_time = time.time()
    check_interval = 60  # Check every minute for faster feedback
    
    while True:
        try:
            elapsed_time = (time.time() - start_time) / 60  # minutes
            
            # Check instance status
            response = ec2_client.describe_instances(InstanceIds=instance_ids)
            
            running_count = 0
            stopped_count = 0
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    state = instance['State']['Name']
                    if state == 'running':
                        running_count += 1
                    elif state in ['stopped', 'terminated']:
                        stopped_count += 1
            
            # Check for completion markers in S3
            try:
                response = s3_client.list_objects_v2(
                    Bucket=s3_bucket,
                    Prefix='results/'
                )
                
                if response.get('Contents'):
                    result_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.pkl')]
                    completion_files = [obj for obj in response['Contents'] if 'COMPLETED_MESA_FREE_INSTANCE_' in obj['Key']]
                    
                    print(f"Progress: {len(result_files)} result files, {len(completion_files)} completed instances")
                    
                    if len(completion_files) >= num_instances:
                        print("SUCCESS All MESA-FREE instances completed successfully!")
                        return True
            except ClientError:
                pass  # S3 check failed, continue monitoring
            
            print(f"Status after {elapsed_time:.1f} minutes: Running={running_count}, Stopped={stopped_count}")
            
            # Safety timeout (much shorter for MESA-FREE version)
            timeout_minutes = 60 if num_instances == 1 else 90  # Reduced timeout
            if elapsed_time > timeout_minutes:
                print(f"Timeout reached after {timeout_minutes} minutes. Checking final results...")
                break
                
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
            break
        except Exception as e:
            print(f"Error monitoring: {e}")
            time.sleep(check_interval)
    
    return False

def download_results(s3_bucket):
    """Download results from S3"""
    
    try:
        s3_client = boto3.client('s3')
        
        # List result files
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix='results/'
        )
        
        if not response.get('Contents'):
            print("ERROR No results found in S3")
            return False
        
        # Download all result files
        os.makedirs('results', exist_ok=True)
        
        result_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.pkl')]
        
        print(f"Downloading {len(result_files)} MESA-FREE result files...")
        
        for obj in result_files:
            key = obj['Key']
            filename = os.path.basename(key)
            local_path = os.path.join('results', filename)
            
            s3_client.download_file(s3_bucket, key, local_path)
            print(f"SUCCESS Downloaded: {filename}")
        
        return len(result_files) > 0
        
    except Exception as e:
        print(f"ERROR Failed to download results: {e}")
        return False

def cleanup_instances(instance_ids):
    """Terminate EC2 instances"""
    
    try:
        ec2_client = boto3.client('ec2')
        ec2_client.terminate_instances(InstanceIds=instance_ids)
        print(f"SUCCESS Terminated {len(instance_ids)} MESA-FREE instances")
    except Exception as e:
        print(f"Could not terminate instances: {e}")

def main():
    """Main deployment function"""
    
    print("ABM Multi-Instance Parameter Sweep - AWS Deployment (MESA-FREE VERSION)")
    print("=" * 80)
    
    # Load configuration
    global config  # Make config global for monitor function
    config = load_config()
    if not config:
        return
    
    num_instances = config['aws_config']['num_instances']
    total_sims = len(config['informality_rates']) * len(config['seeds'])
    
    print(f"MESA-FREE Configuration loaded:")
    print(f"- Total simulations: {total_sims}")
    print(f"- Number of instances: {num_instances}")
    print(f"- Simulations per instance: ~{math.ceil(total_sims/num_instances)}")
    print(f"- Instance type: {config['aws_config']['instance_type']}")
    print(f"- S3 bucket: {config['aws_config']['s3_bucket']}")
    print(f"- MESA DEPENDENCY: REMOVED (much faster!)")
    
    # Calculate costs based on instance type
    instance_costs = {
        't3.micro': 0.0104,
        't3.small': 0.0208,
        't3.medium': 0.0416
    }
    hourly_cost = instance_costs.get(config['aws_config']['instance_type'], 0.0416)
    # Reduce estimated runtime for MESA-free version
    estimated_runtime = config['aws_config']['estimated_runtime_hours'] * 0.5  # 50% faster without Mesa
    estimated_cost = hourly_cost * num_instances * estimated_runtime
    
    print(f"\nEstimated cost (MESA-FREE): ${estimated_cost:.3f}")
    print(f"   ({num_instances} × {config['aws_config']['instance_type']} × {estimated_runtime:.1f} hours)")
    print(f"   (Estimated 50% faster without Mesa dependency)")
    
    confirm = input(f"\nDeploy {total_sims} MESA-FREE simulations across {num_instances} instances? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("Deployment cancelled")
        return
    
    try:
        # Launch instances
        instance_ids = launch_ec2_instances(config)
        if not instance_ids:
            return
        
        print(f"\nMESA-FREE Deployment successful!")
        print(f"Launched instances: {instance_ids}")
        print(f"Estimated completion: {estimated_runtime:.1f} hours (50% faster without Mesa)")
        print(f"Monitor your S3 bucket for results: s3://{config['aws_config']['s3_bucket']}/results/")
        
        # Monitor progress
        if monitor_instances(instance_ids, config['aws_config']['s3_bucket'], num_instances):
            # Download results
            if download_results(config['aws_config']['s3_bucket']):
                print("\nSUCCESS MESA-FREE Results downloaded to 'results/' folder")
                print("Run 'python analyze_results.py' to analyze the results")
        
        # Cleanup
        cleanup_choice = input("\nTerminate all instances? (y/n): ").lower().strip()
        if cleanup_choice == 'y':
            cleanup_instances(instance_ids)
        
        print("\n🎉 MESA-FREE Multi-instance parameter sweep completed!")
        print("This version should be much faster and more reliable!")
        
    except Exception as e:
        print(f"ERROR Deployment failed: {e}")

if __name__ == "__main__":
    main()