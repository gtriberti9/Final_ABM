#!/usr/bin/env python3
"""
AWS Debug Script for ABM Deployment Issues
Helps diagnose why AWS simulations aren't producing results
"""

import boto3
import json
import time
from datetime import datetime
from botocore.exceptions import ClientError

def load_config():
    """Load AWS configuration"""
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ aws_config.json not found")
        return None

def check_s3_bucket(bucket_name):
    """Check S3 bucket and list contents"""
    print(f"\n🔍 Checking S3 bucket: {bucket_name}")
    
    try:
        s3_client = boto3.client('s3')
        
        # List all objects in bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print(f"📁 Bucket is empty")
            return []
        
        objects = response['Contents']
        print(f"📁 Found {len(objects)} objects in bucket:")
        
        deployment_files = []
        result_files = []
        completion_markers = []
        
        for obj in objects:
            key = obj['Key']
            size = obj['Size']
            modified = obj['LastModified']
            
            print(f"   📄 {key} ({size} bytes, {modified})")
            
            if 'deployment' in key:
                deployment_files.append(key)
            elif key.endswith('.pkl'):
                result_files.append(key)
            elif 'COMPLETED' in key:
                completion_markers.append(key)
        
        print(f"\n📊 Summary:")
        print(f"   - Deployment files: {len(deployment_files)}")
        print(f"   - Result files: {len(result_files)}")
        print(f"   - Completion markers: {len(completion_markers)}")
        
        return {
            'deployment_files': deployment_files,
            'result_files': result_files,
            'completion_markers': completion_markers,
            'total_objects': len(objects)
        }
        
    except Exception as e:
        print(f"❌ Error checking S3: {e}")
        return None

def check_running_instances():
    """Check for running EC2 instances"""
    print(f"\n🔍 Checking EC2 instances...")
    
    try:
        ec2_client = boto3.client('ec2')
        
        # Get all instances with ABM tags
        response = ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:Project', 'Values': ['ABM-Multi-Instance', 'ABM-MESA-FREE-Multi-Instance']},
                {'Name': 'instance-state-name', 'Values': ['running', 'pending', 'stopping', 'stopped']}
            ]
        )
        
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append(instance)
        
        if not instances:
            print("📱 No ABM instances found")
            return []
        
        print(f"📱 Found {len(instances)} ABM instances:")
        
        for instance in instances:
            instance_id = instance['InstanceId']
            state = instance['State']['Name']
            instance_type = instance['InstanceType']
            launch_time = instance['LaunchTime']
            
            # Get tags
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            name = tags.get('Name', 'Unknown')
            
            print(f"   🖥️  {instance_id} ({name})")
            print(f"      State: {state}")
            print(f"      Type: {instance_type}")
            print(f"      Launched: {launch_time}")
            
            if state == 'running':
                # Get console output for running instances
                try:
                    console_response = ec2_client.get_console_output(InstanceId=instance_id)
                    output = console_response.get('Output', '')
                    
                    if output:
                        print(f"      Console output preview:")
                        lines = output.split('\n')
                        # Show last 10 lines
                        for line in lines[-10:]:
                            if line.strip():
                                print(f"         {line}")
                    else:
                        print(f"      ⚠️ No console output available yet")
                        
                except Exception as e:
                    print(f"      ⚠️ Could not get console output: {e}")
            
            print()
        
        return instances
        
    except Exception as e:
        print(f"❌ Error checking instances: {e}")
        return []

def get_instance_logs(instance_id):
    """Get detailed logs from a specific instance"""
    print(f"\n🔍 Getting detailed logs for instance: {instance_id}")
    
    try:
        ec2_client = boto3.client('ec2')
        
        # Get console output
        response = ec2_client.get_console_output(InstanceId=instance_id)
        output = response.get('Output', '')
        
        if not output:
            print("❌ No console output available")
            return
        
        print("📋 Console Output:")
        print("=" * 80)
        print(output)
        print("=" * 80)
        
        # Look for specific error patterns
        lines = output.lower().split('\n')
        errors = []
        warnings = []
        
        for line in lines:
            if 'error' in line or 'failed' in line:
                errors.append(line.strip())
            elif 'warning' in line:
                warnings.append(line.strip())
        
        if errors:
            print(f"\n❌ Found {len(errors)} error-related lines:")
            for error in errors[-5:]:  # Show last 5 errors
                print(f"   {error}")
        
        if warnings:
            print(f"\n⚠️ Found {len(warnings)} warning-related lines:")
            for warning in warnings[-3:]:  # Show last 3 warnings
                print(f"   {warning}")
                
    except Exception as e:
        print(f"❌ Error getting logs: {e}")

def check_iam_permissions():
    """Check IAM permissions"""
    print(f"\n🔍 Checking IAM permissions...")
    
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        print(f"🔐 Current AWS identity:")
        print(f"   Account: {identity.get('Account')}")
        print(f"   User/Role: {identity.get('Arn')}")
        
        # Try to list S3 buckets (basic permission test)
        s3_client = boto3.client('s3')
        buckets = s3_client.list_buckets()
        print(f"✅ S3 access: Can list {len(buckets['Buckets'])} buckets")
        
        # Try to describe EC2 instances
        ec2_client = boto3.client('ec2')
        instances = ec2_client.describe_instances()
        print(f"✅ EC2 access: Can describe instances")
        
        return True
        
    except Exception as e:
        print(f"❌ Permission error: {e}")
        return False

def diagnose_common_issues():
    """Diagnose common AWS deployment issues"""
    print(f"\n🔍 Diagnosing common issues...")
    
    issues_found = []
    
    # Check 1: Configuration file
    config = load_config()
    if not config:
        issues_found.append("Missing aws_config.json file")
        return issues_found
    
    # Check 2: S3 bucket
    bucket_name = config['aws_config']['s3_bucket']
    s3_info = check_s3_bucket(bucket_name)
    
    if s3_info is None:
        issues_found.append("Cannot access S3 bucket")
    elif s3_info['total_objects'] == 0:
        issues_found.append("S3 bucket is empty - deployment may not have started")
    elif len(s3_info['result_files']) == 0:
        issues_found.append("No result files found - simulations may be failing")
    
    # Check 3: Running instances
    instances = check_running_instances()
    
    if not instances:
        issues_found.append("No ABM instances found - deployment may have failed")
    else:
        running_count = sum(1 for i in instances if i['State']['Name'] == 'running')
        if running_count == 0:
            issues_found.append("No instances currently running")
    
    # Check 4: IAM permissions
    if not check_iam_permissions():
        issues_found.append("IAM permission issues detected")
    
    return issues_found

def suggest_fixes(issues):
    """Suggest fixes for common issues"""
    print(f"\n💡 SUGGESTED FIXES:")
    print("=" * 50)
    
    if not issues:
        print("🎉 No obvious issues found!")
        print("\nPossible reasons for missing results:")
        print("1. Simulations are still running (check console output)")
        print("2. Simulations are taking longer than expected")
        print("3. Package installation issues on EC2")
        print("4. User data script errors")
        return
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
        
        if "aws_config.json" in issue:
            print("   Fix: Run 'python simple_aws_setup.py' first")
            
        elif "S3 bucket" in issue:
            print("   Fix: Check bucket name and permissions")
            print("   Try: aws s3 ls s3://your-bucket-name")
            
        elif "deployment may not have started" in issue:
            print("   Fix: Run 'python run_aws_sweep.py' to start deployment")
            
        elif "No result files" in issue:
            print("   Fix: Check instance console output for errors")
            print("   Wait longer - simulations may still be running")
            
        elif "No ABM instances" in issue:
            print("   Fix: Check EC2 console for terminated instances")
            print("   Re-run deployment with 'python run_aws_sweep.py'")
            
        elif "No instances currently running" in issue:
            print("   Fix: Instances may have completed and shut down")
            print("   Check S3 for results or restart deployment")
            
        elif "IAM permission" in issue:
            print("   Fix: Ensure proper AWS credentials are configured")
            print("   Try: aws configure list")
        
        print()

def main():
    """Main diagnostic function"""
    print("AWS ABM Deployment Diagnostic Tool")
    print("=" * 50)
    
    # Run diagnostics
    issues = diagnose_common_issues()
    
    # Get user input for detailed instance inspection
    instances = []
    try:
        ec2_client = boto3.client('ec2')
        response = ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:Project', 'Values': ['ABM-Multi-Instance', 'ABM-MESA-FREE-Multi-Instance']},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append(instance['InstanceId'])
                
    except Exception:
        pass
    
    if instances:
        print(f"\n🔍 Found {len(instances)} running instances")
        choice = input("Would you like to see detailed logs for a specific instance? (y/n): ")
        
        if choice.lower() == 'y':
            print("Available instances:")
            for i, instance_id in enumerate(instances):
                print(f"  {i+1}. {instance_id}")
            
            try:
                selection = int(input("Select instance number: ")) - 1
                if 0 <= selection < len(instances):
                    get_instance_logs(instances[selection])
            except (ValueError, IndexError):
                print("Invalid selection")
    
    # Suggest fixes
    suggest_fixes(issues)
    
    print(f"\n📋 DEBUGGING CHECKLIST:")
    print("1. Check /var/log/cloud-init-output.log on EC2 instances")
    print("2. Verify package installation completed successfully")
    print("3. Ensure Python script is executable and has correct permissions")
    print("4. Check if simulations are still running (may take 30-60 minutes)")
    print("5. Monitor S3 bucket for new result files")

if __name__ == "__main__":
    main()