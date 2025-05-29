import boto3
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
zip_file_path = "abm_lambda.zip"  # Update this path
function_name = "abm_parameter_sweep"
handler_name = "lambda_handler.lambda_handler"  
runtime = "python3.9"
region = "us-east-1"
queue_name = "ABMTaskQueue"

# Create AWS clients
aws_lambda = boto3.client('lambda', region_name=region)
iam_client = boto3.client('iam')
sqs_client = boto3.client('sqs', region_name=region)

def get_or_create_queue():
    """Get existing queue URL or create new one"""
    try:
        # Try to find existing queue
        queues = sqs_client.list_queues(QueueNamePrefix=queue_name)
        if "QueueUrls" in queues and queues["QueueUrls"]:
            queue_url = queues["QueueUrls"][0]
            logger.info(f"Found existing queue: {queue_url}")
            return queue_url
    except Exception as e:
        logger.error(f"Error listing queues: {str(e)}")
    
    # Create queue if not exists
    try:
        response = sqs_client.create_queue(
            QueueName=queue_name,
            Attributes={
                'VisibilityTimeout': '300',  # 5 minutes
                'MessageRetentionPeriod': '1209600',  # 14 days
                'ReceiveMessageWaitTimeSeconds': '20'  # Long polling
            }
        )
        queue_url = response["QueueUrl"]
        logger.info(f"Created new queue: {queue_url}")
        return queue_url
    except Exception as e:
        logger.error(f"Failed to create queue: {str(e)}")
        raise

def get_queue_arn(queue_url):
    """Get the ARN of the SQS queue"""
    try:
        response = sqs_client.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn']
        )
        return response['Attributes']['QueueArn']
    except Exception as e:
        logger.error(f"Failed to get queue ARN: {str(e)}")
        raise

def add_sqs_permission(queue_url, queue_arn, lambda_arn):
    """Add permission for Lambda to receive messages from SQS"""
    try:
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sqs:ReceiveMessage",
                    "Resource": queue_arn,
                    "Condition": {
                        "StringEquals": {
                            "aws:SourceArn": lambda_arn
                        }
                    }
                }
            ]
        }
        
        sqs_client.set_queue_attributes(
            QueueUrl=queue_url,
            Attributes={
                'Policy': json.dumps(policy)
            }
        )
        logger.info("Added SQS permissions for Lambda")
    except Exception as e:
        logger.warning(f"Failed to set SQS policy (might already exist): {str(e)}")

def create_or_update_lambda_function():
    """Create or update the Lambda function"""
    # Get the role ARN
    role_name = "LabRole"  # Your actual Lambda execution role name
    role = iam_client.get_role(RoleName=role_name)
    
    with open(zip_file_path, 'rb') as zip_file:
        lambda_zip = zip_file.read()

    try:
        response = aws_lambda.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role['Role']['Arn'],
            Handler=handler_name,
            Code=dict(ZipFile=lambda_zip),
            Timeout=300,  # 5 minutes - increase if needed
            MemorySize=1024,  # Adjust as needed
        )
        logger.info(f"Lambda function {function_name} created successfully.")
        lambda_arn = response['FunctionArn']
        
        # Set concurrency limit separately
        set_concurrency_limit(function_name)
        return lambda_arn

    except aws_lambda.exceptions.ResourceConflictException:
        # If function already exists, update it
        response = aws_lambda.update_function_code(
            FunctionName=function_name,
            ZipFile=lambda_zip,
        )
        logger.info(f"Lambda function {function_name} updated successfully.")
        
        # Set concurrency limit
        set_concurrency_limit(function_name)
        
        # Get function ARN
        func_response = aws_lambda.get_function(FunctionName=function_name)
        return func_response['Configuration']['FunctionArn']

def set_concurrency_limit(function_name, limit=50):
    """Set reserved concurrency limit for the Lambda function"""
    try:
        # Correct method name is put_function_concurrency
        response = aws_lambda.put_function_concurrency(
            FunctionName=function_name,
            ReservedConcurrentExecutions=limit
        )
        logger.info(f"Set reserved concurrency limit to {limit} for {function_name}")
    except Exception as e:
        logger.warning(f"Could not set concurrency limit: {str(e)}")
        logger.info("Function will use default auto-scaling behavior (this is fine for testing)")

def create_sqs_trigger(lambda_arn, queue_arn):
    """Create SQS trigger for Lambda function"""
    try:
        # Check if event source mapping already exists
        existing_mappings = aws_lambda.list_event_source_mappings(
            FunctionName=function_name,
            EventSourceArn=queue_arn
        )
        
        if existing_mappings['EventSourceMappings']:
            logger.info("SQS trigger already exists")
            return existing_mappings['EventSourceMappings'][0]['UUID']
        
        # Create new event source mapping (simplified parameters)
        response = aws_lambda.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=function_name,
            BatchSize=1,  # Process one message at a time
            MaximumBatchingWindowInSeconds=0  # No batching delay
            # Removed ReportBatchItemFailures - not supported in older boto3
        )
        
        logger.info(f"Created SQS trigger with UUID: {response['UUID']}")
        return response['UUID']
        
    except Exception as e:
        logger.error(f"Failed to create SQS trigger: {str(e)}")
        raise

def main():
    """Main deployment function"""
    try:
        # Step 1: Create or get SQS queue
        logger.info("Step 1: Setting up SQS queue...")
        queue_url = get_or_create_queue()
        queue_arn = get_queue_arn(queue_url)
        
        # Step 2: Create or update Lambda function
        logger.info("Step 2: Deploying Lambda function...")
        lambda_arn = create_or_update_lambda_function()
        
        # Step 3: Add SQS permissions for Lambda
        logger.info("Step 3: Setting SQS permissions...")
        add_sqs_permission(queue_url, queue_arn, lambda_arn)
        
        # Step 4: Create SQS trigger
        logger.info("Step 4: Creating SQS trigger...")
        trigger_uuid = create_sqs_trigger(lambda_arn, queue_arn)
        
        logger.info("=" * 50)
        logger.info("DEPLOYMENT SUCCESSFUL!")
        logger.info(f"Lambda Function: {function_name}")
        logger.info(f"SQS Queue: {queue_name}")
        logger.info(f"Queue URL: {queue_url}")
        logger.info(f"Trigger UUID: {trigger_uuid}")
        logger.info("=" * 50)
        
        return {
            'lambda_arn': lambda_arn,
            'queue_url': queue_url,
            'queue_arn': queue_arn,
            'trigger_uuid': trigger_uuid
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()