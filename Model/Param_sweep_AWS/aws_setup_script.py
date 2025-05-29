#!/usr/bin/env python3
"""
AWS Setup Automation Script for ABM Parameter Sweep
"""

import boto3
import json
import time
import random
import string
from botocore.exceptions import ClientError, NoCredentialsError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AWSSetupAutomation:
    """Automate AWS setup for ABM parameter sweep"""
    
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        try:
            self.ec2_client = boto3.client('ec2', region_name=region_name)
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.iam_client = boto3.client('iam', region_name=region_name)
            self.sts_client = boto3.client('sts', region_name=region_name)
            logger.info(f"AWS clients initialized for region: {region_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please run 'aws configure' first.")
            raise
    
    def test_aws_connection(self):
        """Test AWS connection and permissions"""
        try:
            # Test credentials
            identity = self.sts_client.get_caller_identity()
            logger.info(f"AWS connection successful. Account: {identity.get('Account')}")
            
            # Test S3 access
            self.s3_client.list_buckets()
            logger.info("S3 access verified")
            
            # Test EC2 access
            self.ec2_client.describe_regions()
            logger.info("EC2 access verified")
            
            return True
            
        except Exception as e:
            logger.error(f"AWS connection test failed: {str(e)}")
            return False
    
    def create_s3_bucket(self, bucket_name=None):
        """Create S3 bucket for results storage"""
        
        if not bucket_name:
            # Generate unique bucket name
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            bucket_name = f"abm-sweep-results-{random_suffix}"
        
        try:
            if self.region_name == 'us-east-1':
                # us-east-1 doesn't need LocationConstraint
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region_name}
                )
            
            logger.info(f"S3 bucket created: {bucket_name}")
            return bucket_name
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyExists':
                logger.warning(f"Bucket {bucket_name} already exists")
                return bucket_name
            else:
                logger.error(f"Failed to create bucket: {str(e)}")
                return None
    
    def create_iam_role(self):
        """Create IAM role for EC2 instances"""
        
        role_name = "EC2-S3-Access-Role"
        
        # Trust policy for EC2
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Create role
            self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Role for ABM parameter sweep EC2 instances"
            )
            logger.info(f"IAM role created: {role_name}")
            
            # Attach S3 access policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
            )
            
            # Attach CloudWatch logs policy (optional but helpful)
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
            )
            
            # Create instance profile
            self.iam_client.create_instance_profile(InstanceProfileName=role_name)
            
            # Add role to instance profile
            self.iam_client.add_role_to_instance_profile(
                InstanceProfileName=role_name,
                RoleName=role_name
            )
            
            logger.info("IAM role and instance profile configured")
            
            # Wait for role to propagate
            logger.info("Waiting for IAM role to propagate (30 seconds)...")
            time.sleep(30)
            
            return role_name
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                logger.warning(f"IAM role {role_name} already exists")
                return role_name
            else:
                logger.error(f"Failed to create IAM role: {str(e)}")
                return None
    
    def create_security_group(self):
        """Create security group for EC2 instances"""
        
        group_name = "abm-sweep-sg"
        
        try:
            response = self.ec2_client.create_security_group(
                GroupName=group_name,
                Description="Security group for ABM parameter sweep"
            )
            
            security_group_id = response['GroupId']
            logger.info(f"Security group created: {security_group_id}")
            
            # Add outbound internet access (usually allowed by default)
            # Add inbound SSH access (optional)
            try:
                self.ec2_client.authorize_security_group_ingress(
                    GroupId=security_group_id,
                    IpPermissions=[
                        {
                            'IpProtocol': 'tcp',
                            'FromPort': 22,
                            'ToPort': 22,
                            'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                        }
                    ]
                )
                logger.info("SSH access added to security group")
            except ClientError:
                logger.info("SSH rule already exists or couldn't be added")
            
            return security_group_id
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
                # Get existing security group ID
                response = self.ec2_client.describe_security_groups(GroupNames=[group_name])
                security_group_id = response['SecurityGroups'][0]['GroupId']
                logger.warning(f"Security group {group_name} already exists: {security_group_id}")
                return security_group_id
            else:
                logger.error(f"Failed to create security group: {str(e)}")
                return None
    
    def create_key_pair(self, key_name="abm-sweep-key"):
        """Create EC2 key pair"""
        
        try:
            response = self.ec2_client.create_key_pair(KeyName=key_name)
            
            # Save private key to file
            with open(f"{key_name}.pem", 'w') as f:
                f.write(response['KeyMaterial'])
            
            # Set proper permissions on Unix systems
            import os
            import stat
            os.chmod(f"{key_name}.pem", stat.S_IRUSR | stat.S_IWUSR)
            
            logger.info(f"Key pair created: {key_name} (saved as {key_name}.pem)")
            return key_name
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
                logger.warning(f"Key pair {key_name} already exists")
                return key_name
            else:
                logger.error(f"Failed to create key pair: {str(e)}")
                return None
    
    def setup_complete_aws_environment(self):
        """Set up complete AWS environment for ABM sweep"""
        
        logger.info("Starting complete AWS environment setup...")
        
        # Test connection
        if not self.test_aws_connection():
            return None
        
        setup_results = {
            'region': self.region_name,
            'success': True
        }
        
        # Create S3 bucket
        bucket_name = self.create_s3_bucket()
        if bucket_name:
            setup_results['s3_bucket'] = bucket_name
        else:
            setup_results['success'] = False
            return setup_results
        
        # Create IAM role
        role_name = self.create_iam_role()
        if role_name:
            setup_results['iam_role'] = role_name
        else:
            setup_results['success'] = False
            return setup_results
        
        # Create security group
        security_group_id = self.create_security_group()
        if security_group_id:
            setup_results['security_group_id'] = security_group_id
        
        # Create key pair
        key_name = self.create_key_pair()
        if key_name:
            setup_results['key_pair_name'] = key_name
        
        logger.info("AWS environment setup completed successfully!")
        return setup_results
    
    def generate_config_file(self, setup_results):
        """Generate configuration file for ABM sweep"""
        
        config = {
            "mode": "aws",
            "informality_rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "seeds": list(range(100)),
            "auto_analyze": True,
            "auto_cleanup": True,
            "model_params": {
                "n_firms": 50,
                "n_consumers": 200,
                "n_banks": 5,
                "max_steps": 200,
                "inflation_target": 0.02,
                "initial_policy_rate": 0.03,
                "current_inflation": 0.02
            },
            "aws_config": {
                "region_name": setup_results['region'],
                "instance_type": "c5.2xlarge",
                "num_instances": 4,
                "s3_bucket": setup_results['s3_bucket'],
                "key_pair_name": setup_results.get('key_pair_name'),
                "security_group_id": setup_results.get('security_group_id'),
                "estimated_runtime_hours": 3.0,
                "auto_confirm": False
            }
        }
        
        # Save configuration
        with open('aws_sweep_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Configuration file saved: aws_sweep_config.json")
        return config

def main():
    """Main setup function"""
    
    print("ABM Parameter Sweep - AWS Environment Setup")
    print("="*50)
    
    # Get user preferences
    region = input("Enter AWS region [us-east-1]: ").strip() or 'us-east-1'
    
    try:
        # Initialize setup
        setup = AWSSetupAutomation(region_name=region)
        
        # Run complete setup
        results = setup.setup_complete_aws_environment()
        
        if results and results['success']:
            print(f"\n✅ AWS Environment Setup Complete!")
            print(f"Region: {results['region']}")
            print(f"S3 Bucket: {results.get('s3_bucket')}")
            print(f"IAM Role: {results.get('iam_role')}")
            print(f"Security Group: {results.get('security_group_id')}")
            print(f"Key Pair: {results.get('key_pair_name')}")
            
            # Generate config file
            config = setup.generate_config_file(results)
            
            print(f"\n📋 Configuration file created: aws_sweep_config.json")
            print(f"\nYou can now run the full parameter sweep with:")
            print(f"python main_runner.py --config aws_sweep_config.json")
            
            # Estimate costs
            print(f"\n💰 Estimated Cost for Full Sweep (800 simulations):")
            print(f"- Instance Type: c5.2xlarge ($0.34/hour)")
            print(f"- Number of Instances: 4") 
            print(f"- Estimated Runtime: 3 hours")
            print(f"- Total Estimated Cost: ~$4.08")
            
        else:
            print("❌ AWS setup failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        print("Please ensure you have run 'aws configure' and have proper permissions.")

if __name__ == "__main__":
    main()