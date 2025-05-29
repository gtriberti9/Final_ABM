import boto3
import time
import matplotlib.pyplot as plt
import json
import logging
from deploy_lambda import main as deploy_infrastructure  # Import deployment function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REGION = "us-east-1"
TABLE_NAME = "ABMResults"
FUNCTION_NAME = "abm_parameter_sweep"

informality_rates = [0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
seeds = range(100)

def ensure_table_exists(table_name, region="us-east-1"):
    """Create DynamoDB table if it doesn't exist"""
    dynamodb = boto3.client("dynamodb", region_name=region)
    try:
        dynamodb.describe_table(TableName=table_name)
        logger.info(f"Table '{table_name}' already exists.")
    except dynamodb.exceptions.ResourceNotFoundException:
        logger.info(f"Table '{table_name}' not found. Creating...")
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "informality_rate", "KeyType": "HASH"},
                {"AttributeName": "seed", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "informality_rate", "AttributeType": "N"},
                {"AttributeName": "seed", "AttributeType": "N"},
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 10, "WriteCapacityUnits": 10}
        )
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        logger.info(f"Table '{table_name}' created and ready.")

def clear_dynamodb_table(table_name):
    """Clear all items from DynamoDB table"""
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)
    scan = table.scan(ProjectionExpression="#k,#s", ExpressionAttributeNames={"#k": "informality_rate", "#s": "seed"})
    with table.batch_writer() as batch:
        for item in scan["Items"]:
            batch.delete_item(Key=item)
    while 'LastEvaluatedKey' in scan:
        scan = table.scan(
            ProjectionExpression="#k,#s",
            ExpressionAttributeNames={"#k": "informality_rate", "#s": "seed"},
            ExclusiveStartKey=scan['LastEvaluatedKey']
        )
        with table.batch_writer() as batch:
            for item in scan["Items"]:
                batch.delete_item(Key=item)
    logger.info(f"Cleared table {table_name}")

def fill_sqs_tasks(queue_url):
    """Fill SQS queue with simulation tasks"""
    sqs = boto3.client("sqs", region_name=REGION)
    # Clear any existing messages first
    try:
        while True:
            msgs = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
            if "Messages" not in msgs:
                break
            for msg in msgs["Messages"]:
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
    except:
        pass
    
    # Add new tasks
    total_tasks = 0
    for rate in informality_rates:
        for seed in seeds:
            msg = json.dumps({"informality_rate": rate, "seed": seed})
            try:
                sqs.send_message(QueueUrl=queue_url, MessageBody=msg)
                total_tasks += 1
            except Exception as e:
                logger.error(f"Failed to send message for rate={rate}, seed={seed}: {str(e)}")
    logger.info(f"Added {total_tasks} tasks to SQS queue.")
    return total_tasks

def wait_for_completion(table_name, expected_count, timeout_minutes=60):
    """Wait for all simulations to complete"""
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)
    start_time = time.time()
    
    while True:
        try:
            response = table.scan(Select='COUNT')
            current_count = response['Count']
            elapsed_minutes = (time.time() - start_time) / 60
            logger.info(f"Current DynamoDB item count: {current_count}/{expected_count} (elapsed: {elapsed_minutes:.1f} min)")
            
            if current_count >= expected_count:
                logger.info("All tasks completed!")
                break
            if elapsed_minutes > timeout_minutes:
                logger.warning(f"Timeout reached after {timeout_minutes} minutes. Current count: {current_count}")
                break
            time.sleep(30)
        except Exception as e:
            logger.error(f"Error checking completion status: {str(e)}")
            time.sleep(30)

def check_lambda_health():
    """Check if Lambda function is healthy and ready"""
    lambda_client = boto3.client('lambda', region_name=REGION)
    try:
        response = lambda_client.get_function(FunctionName=FUNCTION_NAME)
        state = response['Configuration']['State']
        logger.info(f"Lambda function state: {state}")
        return state == 'Active'
    except Exception as e:
        logger.error(f"Error checking Lambda function: {str(e)}")
        return False

def monitor_sqs_and_lambda(queue_url, expected_tasks):
    """Monitor SQS queue and Lambda function metrics"""
    sqs = boto3.client("sqs", region_name=REGION)
    cloudwatch = boto3.client('cloudwatch', region_name=REGION)
    
    try:
        # Check queue attributes
        queue_attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
        )
        
        messages_available = int(queue_attrs['Attributes'].get('ApproximateNumberOfMessages', 0))
        messages_in_flight = int(queue_attrs['Attributes'].get('ApproximateNumberOfMessagesNotVisible', 0))
        
        logger.info(f"SQS Status - Available: {messages_available}, In Flight: {messages_in_flight}")
        
        # Get Lambda metrics from CloudWatch
        try:
            lambda_metrics = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': FUNCTION_NAME}],
                StartTime=time.time() - 600,  # Last 10 minutes
                EndTime=time.time(),
                Period=300,
                Statistics=['Sum']
            )
            
            total_invocations = sum([point['Sum'] for point in lambda_metrics['Datapoints']])
            logger.info(f"Lambda invocations in last 10 min: {total_invocations}")
            
        except Exception as e:
            logger.warning(f"Could not get Lambda metrics: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error monitoring SQS/Lambda: {str(e)}")

def run_experiment(deployment_info):
    """Run the complete experiment"""
    try:
        # Step 1: Ensure DynamoDB table exists
        logger.info("Step 1: Setting up DynamoDB table...")
        ensure_table_exists(TABLE_NAME, region=REGION)
        clear_dynamodb_table(TABLE_NAME)
        
        # Step 2: Check Lambda health
        logger.info("Step 2: Checking Lambda function health...")
        if not check_lambda_health():
            logger.error("Lambda function is not healthy. Please check deployment.")
            return None
        
        # Step 3: Fill SQS with tasks
        logger.info("Step 3: Queueing simulation tasks...")
        queue_url = deployment_info['queue_url']
        actual_tasks = fill_sqs_tasks(queue_url)
        expected_results = len(informality_rates) * len(seeds)
        logger.info(f"Expected results: {expected_results}, Actual tasks queued: {actual_tasks}")
        
        # Step 4: Monitor progress
        logger.info("Step 4: Monitoring execution...")
        start_time = time.time()
        
        # Monitor every 2 minutes for the first 10 minutes
        for i in range(5):
            time.sleep(120)  # Wait 2 minutes
            monitor_sqs_and_lambda(queue_url, actual_tasks)
        
        # Step 5: Wait for completion
        logger.info("Step 5: Waiting for completion...")
        wait_for_completion(TABLE_NAME, min(expected_results, actual_tasks), timeout_minutes=90)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Experiment completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        return total_time
        
    except Exception as e:
        logger.error(f"Error in experiment: {str(e)}")
        raise

def create_performance_plots(execution_time):
    """Create performance visualization plots"""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Execution time
    plt.subplot(2, 2, 1)
    plt.bar(['Lambda Parallel Execution'], [execution_time/60])
    plt.ylabel('Time (minutes)')
    plt.title('Total Execution Time')
    
    # Plot 2: Theoretical speedup
    sequential_estimate = len(informality_rates) * len(seeds) * 30  # 30 seconds per simulation
    speedup = sequential_estimate / execution_time
    plt.subplot(2, 2, 2)
    plt.bar(['Speedup Factor'], [speedup])
    plt.ylabel('Speedup Factor')
    plt.title(f'Parallel Speedup vs Sequential\n(Est. {speedup:.1f}x faster)')
    
    # Plot 3: Cost efficiency (tasks per minute)
    tasks_per_minute = (len(informality_rates) * len(seeds)) / (execution_time / 60)
    plt.subplot(2, 2, 3)
    plt.bar(['Tasks/Minute'], [tasks_per_minute])
    plt.ylabel('Tasks per Minute')
    plt.title('Processing Rate')
    
    # Plot 4: Parameter space coverage
    plt.subplot(2, 2, 4)
    plt.bar(['Parameter Combinations'], [len(informality_rates) * len(seeds)])
    plt.ylabel('Number of Simulations')
    plt.title('Parameter Space Coverage')
    
    plt.tight_layout()
    plt.savefig("experiment_performance.png", dpi=300, bbox_inches='tight')
    logger.info("Performance plots saved as 'experiment_performance.png'")
    plt.show()

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STARTING ABM PARAMETER SWEEP EXPERIMENT")
    logger.info("=" * 60)
    
    try:
        # Step 1: Deploy infrastructure
        logger.info("Phase 1: Deploying AWS infrastructure...")
        deployment_info = deploy_infrastructure()
        
        # Step 2: Run experiment
        logger.info("Phase 2: Running parameter sweep experiment...")
        execution_time = run_experiment(deployment_info)
        
        if execution_time:
            # Step 3: Create performance analysis
            logger.info("Phase 3: Creating performance analysis...")
            create_performance_plots(execution_time)
            
            # Step 4: Summary report
            logger.info("=" * 60)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {execution_time/60:.1f} minutes")
            logger.info(f"Parameter combinations tested: {len(informality_rates)} × {len(seeds)} = {len(informality_rates) * len(seeds)}")
            logger.info(f"Average time per simulation: {execution_time/(len(informality_rates) * len(seeds)):.1f} seconds")
            logger.info(f"Results stored in DynamoDB table: {TABLE_NAME}")
            logger.info("=" * 60)
        else:
            logger.error("Experiment failed to complete successfully")
            
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()