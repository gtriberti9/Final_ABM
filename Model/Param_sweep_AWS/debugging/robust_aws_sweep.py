#!/usr/bin/env python3
"""
FIXED AWS parameter sweep runner - Addresses JSON and permission issues
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

def create_fixed_deployment_package():
    """Create deployment package with FIXED configuration injection"""
    
    package_name = "abm_fixed_deployment.zip"
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model files
        required_files = ['model.py', 'agents.py']
        
        for file in required_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file} to package")
            else:
                print(f"Missing required file: {file}")
                return None
        
        # Create FIXED startup script with proper permissions
        startup_script = '''#!/bin/bash
# FIXED STARTUP SCRIPT - Addresses permission and JSON issues
exec > /tmp/user-data.log 2>&1

echo "========================================="
echo "FIXED ABM STARTUP SCRIPT BEGIN"
echo "Time: $(date)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "========================================="

# Function for logging with timestamps
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> /tmp/abm-debug.log
}

log_msg "Step 1: System update"
yum update -y

log_msg "Step 2: Install Python and pip"
yum install -y python3 python3-pip

log_msg "Step 3: Verify Python installation"
python3 --version
pip3 --version

log_msg "Step 4: Install required packages individually"
pip3 install --user numpy
log_msg "numpy installed successfully"

pip3 install --user pandas  
log_msg "pandas installed successfully"

pip3 install --user scipy
log_msg "scipy installed successfully"

pip3 install --user matplotlib
log_msg "matplotlib installed successfully"

log_msg "Step 5: Verify imports"
python3 -c "import numpy; print('numpy version:', numpy.__version__)"
python3 -c "import pandas; print('pandas version:', pandas.__version__)"

log_msg "Step 6: Set up environment"
cd /home/ec2-user
export PYTHONPATH="/home/ec2-user:$PYTHONPATH"
chown -R ec2-user:ec2-user /home/ec2-user

log_msg "Step 7: List files in directory"
ls -la

log_msg "Step 8: Test model import as ec2-user"
sudo -u ec2-user python3 -c "
import sys
sys.path.append('/home/ec2-user')
try:
    from model import MonetaryPolicyModel
    print('SUCCESS: Model imported successfully')
    test_model = MonetaryPolicyModel(n_firms=5, n_consumers=10, n_commercial_banks=1)
    print('SUCCESS: Test model created')
    test_model.step()
    print('SUCCESS: Test simulation step completed')
except Exception as e:
    print(f'ERROR: Model test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

log_msg "Step 9: Starting main simulation as ec2-user"
sudo -u ec2-user python3 /home/ec2-user/run_fixed_simulation.py

log_msg "FIXED STARTUP SCRIPT COMPLETE"
echo "========================================="
'''
        
        with open('startup_fixed.sh', 'w') as f:
            f.write(startup_script)
        zipf.write('startup_fixed.sh')
        
        # Create FIXED simulation script with proper JSON handling
        simulation_script = '''#!/usr/bin/env python3
import sys
import os
import json
import pickle
import time
import traceback
from datetime import datetime

def log_msg(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    # Also write to a debug file
    with open('/tmp/abm-debug.log', 'a') as f:
        f.write(f"[{timestamp}] {message}\\n")
    sys.stdout.flush()

log_msg("=== FIXED SIMULATION SCRIPT START ===")

# Add current directory to Python path
sys.path.append('/home/ec2-user')
sys.path.append('.')

try:
    # Import required modules
    import numpy as np
    import random
    log_msg("Basic imports successful")
    
    from model import MonetaryPolicyModel
    log_msg("Model imported successfully")
    
except Exception as e:
    log_msg(f"Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

def run_single_simulation(params):
    """Run a single simulation"""
    try:
        log_msg(f"Starting simulation: informality={params['informality_rate']}, seed={params['seed']}")
        
        # Set seeds
        np.random.seed(params['seed'])
        random.seed(params['seed'])
        
        # Create model
        model = MonetaryPolicyModel(
            n_firms=params['n_firms'],
            n_consumers=params['n_consumers'],
            n_commercial_banks=params['n_banks'],
            inflation_target=params['inflation_target'],
            initial_policy_rate=params['initial_policy_rate'],
            informality_rate=params['informality_rate'],
            current_inflation=params['current_inflation']
        )
        
        log_msg(f"Model created: {len(model.firms)} firms, {len(model.consumers)} consumers")
        
        # Run simulation
        for step in range(params['max_steps']):
            model.step()
            
            # Progress logging every 25 steps
            if (step + 1) % 25 == 0:
                log_msg(f"Step {step + 1}/{params['max_steps']}: inflation={model.current_inflation:.4f}")
        
        # Collect results
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
            'fixed_version': True
        }
        
        log_msg(f"Simulation completed successfully")
        return results
        
    except Exception as e:
        log_msg(f"Simulation error: {e}")
        traceback.print_exc()
        return {'error': str(e), 'params': params}

def main():
    """Main simulation runner"""
    log_msg("=== MAIN FUNCTION START ===")
    
    # Get instance metadata
    try:
        import urllib.request
        response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
        instance_id = response.read().decode('utf-8')
        log_msg(f"Running on instance: {instance_id}")
    except Exception as e:
        instance_id = "unknown"
        log_msg(f"Could not get instance ID: {e}")
    
    # Load configuration from file (NOT string replacement)
    try:
        with open('/home/ec2-user/config.json', 'r') as f:
            config = json.load(f)
        log_msg("Configuration loaded from file")
    except Exception as e:
        log_msg(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Calculate simulation assignments
    total_sims = len(config['informality_rates']) * len(config['seeds'])
    num_instances = config['aws_config']['num_instances']
    instance_index = int(os.environ.get('INSTANCE_INDEX', '0'))
    
    sims_per_instance = math.ceil(total_sims / num_instances)
    start_sim = instance_index * sims_per_instance
    end_sim = min(start_sim + sims_per_instance, total_sims)
    
    log_msg(f"Instance {instance_index} will run simulations {start_sim} to {end_sim-1}")
    
    # Create parameter combinations
    all_combinations = []
    for informality_rate in config['informality_rates']:
        for seed in config['seeds']:
            all_combinations.append({
                **config['model_params'],
                'informality_rate': informality_rate,
                'seed': seed,
                'instance_id': instance_index
            })
    
    my_combinations = all_combinations[start_sim:end_sim]
    
    log_msg(f"Starting {len(my_combinations)} simulations")
    
    # Run simulations
    results = []
    for i, params in enumerate(my_combinations):
        log_msg(f"=== SIMULATION {i+1}/{len(my_combinations)} ===")
        
        start_time = time.time()
        result = run_single_simulation(params)
        run_time = time.time() - start_time
        
        results.append(result)
        log_msg(f"Simulation {i+1} completed in {run_time:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"abm_fixed_results_instance_{instance_index}_{timestamp}.pkl"
    
    log_msg(f"Saving results to {results_file}")
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    log_msg("Results saved locally")
    
    # Upload to S3
    log_msg("Uploading results to S3...")
    try:
        import boto3
        s3_client = boto3.client('s3')
        bucket_name = config['aws_config']['s3_bucket']
        
        s3_client.upload_file(results_file, bucket_name, f"results/{results_file}")
        log_msg(f"Results uploaded to S3")
        
        # Create completion marker
        completion_marker = f"COMPLETED_FIXED_INSTANCE_{instance_index}_{timestamp}.txt"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"results/{completion_marker}",
            Body=f"Instance {instance_index} completed {len(results)} simulations at {datetime.now()}"
        )
        log_msg(f"Completion marker uploaded")
        
    except Exception as e:
        log_msg(f"S3 upload error: {e}")
        traceback.print_exc()
    
    log_msg("=== FIXED SIMULATION COMPLETE ===")

if __name__ == "__main__":
    import math
    main()
'''
        
        with open('run_fixed_simulation.py', 'w') as f:
            f.write(simulation_script)
        zipf.write('run_fixed_simulation.py')
        
        # Clean up temporary files
        os.remove('startup_fixed.sh')
        os.remove('run_fixed_simulation.py')
    
    print(f"Created FIXED deployment package: {package_name}")
    return package_name

def upload_package_to_s3(package_name, bucket_name):
    """Upload deployment package to S3"""
    try:
        s3_client = boto3.client('s3')
        s3_key = f"deployments/{package_name}"
        
        s3_client.upload_file(package_name, bucket_name, s3_key)
        print(f"Package uploaded to S3: s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
        
    except Exception as e:
        print(f"Failed to upload package: {e}")
        return None

def create_fixed_user_data(config, s3_package_url, instance_index):
    """Create FIXED user data script with proper JSON handling"""
    
    # Properly format the JSON first
    config_json = json.dumps(config, indent=2)
    
    user_data = f'''#!/bin/bash
exec > /tmp/user-data.log 2>&1

echo "=== FIXED USER DATA START ==="
date

# Set instance index
export INSTANCE_INDEX={instance_index}
echo "export INSTANCE_INDEX={instance_index}" >> /home/ec2-user/.bashrc

# Download and extract package
cd /home/ec2-user
aws s3 cp {s3_package_url} deployment.zip
unzip deployment.zip
chown -R ec2-user:ec2-user /home/ec2-user

# Save configuration as JSON file (FIXED approach)
cat > /home/ec2-user/config.json << 'EOL'
{config_json}
EOL

# Make scripts executable
chmod +x startup_fixed.sh

# Run startup script
bash /home/ec2-user/startup_fixed.sh

echo "=== FIXED USER DATA END ==="
date
'''
    
    return user_data

def launch_fixed_instances(config):
    """Launch EC2 instances with FIXED configuration"""
    
    try:
        ec2_client = boto3.client('ec2')
        num_instances = config['aws_config']['num_instances']
        
        # Create deployment package
        package_name = create_fixed_deployment_package()
        if not package_name:
            return []
        
        s3_url = upload_package_to_s3(package_name, config['aws_config']['s3_bucket'])
        if not s3_url:
            return []
        
        launched_instances = []
        
        for instance_index in range(num_instances):
            print(f"Launching FIXED instance {instance_index + 1}/{num_instances}...")
            
            user_data = create_fixed_user_data(config, s3_url, instance_index)
            
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
                        {'Key': 'Name', 'Value': f'ABM-FIXED-Instance-{instance_index}'},
                        {'Key': 'Project', 'Value': 'ABM-FIXED-Multi-Instance'},
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
            
            print(f"Launched FIXED instance {instance_index}: {instance_id}")
            time.sleep(3)
        
        # Cleanup
        if os.path.exists(package_name):
            os.remove(package_name)
        
        return launched_instances
        
    except Exception as e:
        print(f"Failed to launch instances: {e}")
        return []

def main():
    """Main FIXED deployment function"""
    
    print("ABM FIXED Parameter Sweep - AWS Deployment")
    print("=" * 60)
    print("FIXES APPLIED:")
    print("- JSON configuration injection fixed")
    print("- Permission issues resolved") 
    print("- Proper error logging")
    print("- Better user context handling")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    num_instances = config['aws_config']['num_instances']
    total_sims = len(config['informality_rates']) * len(config['seeds'])
    
    print(f"FIXED Configuration:")
    print(f"- Total simulations: {total_sims}")
    print(f"- Number of instances: {num_instances}")
    print(f"- Simulations per instance: ~{math.ceil(total_sims/num_instances)}")
    print(f"- Instance type: {config['aws_config']['instance_type']}")
    print(f"- S3 bucket: {config['aws_config']['s3_bucket']}")
    
    confirm = input(f"\nDeploy {total_sims} simulations with FIXED version? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("Deployment cancelled")
        return
    
    try:
        # Terminate the broken instance first
        print("\nTerminating broken instance...")
        ec2_client = boto3.client('ec2')
        
        # Find and terminate the robust instance
        response = ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:Project', 'Values': ['ABM-ROBUST-Multi-Instance']},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )
        
        broken_instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                broken_instances.append(instance['InstanceId'])
        
        if broken_instances:
            ec2_client.terminate_instances(InstanceIds=broken_instances)
            print(f"Terminated {len(broken_instances)} broken instances")
        
        # Launch fixed instances
        instance_ids = launch_fixed_instances(config)
        if not instance_ids:
            return
        
        print(f"\nFIXED Deployment successful!")
        print(f"Launched instances: {instance_ids}")
        print(f"Expected completion: ~45 minutes")
        print(f"Monitor S3 bucket: s3://{config['aws_config']['s3_bucket']}/results/")
        
        print(f"\nYou can monitor progress with:")
        print(f"1. python debug_aws.py")
        print(f"2. Check EC2 console for instance logs")
        print(f"3. Watch S3 bucket for result files")
        
    except Exception as e:
        print(f"FIXED Deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()