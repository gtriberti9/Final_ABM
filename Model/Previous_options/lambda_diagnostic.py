import boto3
import json
from datetime import datetime, timedelta

# AWS clients
logs_client = boto3.client('logs', region_name='us-east-1')
lambda_client = boto3.client('lambda', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
sqs = boto3.client('sqs', region_name='us-east-1')

function_name = "abm_parameter_sweep"
table_name = "ABMResults"

def check_lambda_errors():
    """Check recent Lambda errors"""
    print("🔍 Checking Lambda errors...")
    
    log_group = f"/aws/lambda/{function_name}"
    
    try:
        # Get recent log streams
        streams = logs_client.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )
        
        print(f"Found {len(streams['logStreams'])} recent log streams")
        
        # Check recent logs for errors
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=30)
        
        events = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            filterPattern="ERROR"
        )
        
        if events['events']:
            print(f"❌ Found {len(events['events'])} ERROR events:")
            for event in events['events'][-10:]:  # Show last 10 errors
                print(f"  {event['message']}")
        else:
            print("✅ No ERROR events found")
            
        # Check for any logs at all
        all_events = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=5
        )
        
        if all_events['events']:
            print(f"📝 Recent log samples:")
            for event in all_events['events'][-5:]:
                print(f"  {event['message'][:100]}...")
        else:
            print("⚠️  No recent log events found")
            
    except Exception as e:
        print(f"❌ Error checking logs: {str(e)}")

def check_dynamodb_table():
    """Check DynamoDB table status"""
    print("\n📊 Checking DynamoDB table...")
    
    try:
        table = dynamodb.Table(table_name)
        response = table.scan(Select='COUNT')
        print(f"DynamoDB item count: {response['Count']}")
        
        # Check table status
        table_info = table.Table()
        print(f"Table status: {table_info.table_status}")
        print(f"Item count (table): {table_info.item_count}")
        
        # Try to scan for any items
        sample = table.scan(Limit=5)
        if sample['Items']:
            print(f"Sample items found: {len(sample['Items'])}")
            print(f"Sample item keys: {list(sample['Items'][0].keys())}")
        else:
            print("No items found in table")
            
    except Exception as e:
        print(f"❌ Error checking DynamoDB: {str(e)}")

def check_lambda_permissions():
    """Check Lambda permissions"""
    print("\n🔐 Checking Lambda permissions...")
    
    try:
        # Get function configuration
        response = lambda_client.get_function(FunctionName=function_name)
        role_arn = response['Configuration']['Role']
        print(f"Lambda role ARN: {role_arn}")
        
        # Check if function has DynamoDB permissions
        print("Function should have DynamoDB permissions via LabRole")
        
    except Exception as e:
        print(f"❌ Error checking permissions: {str(e)}")

def test_single_invocation():
    """Test a single Lambda invocation"""
    print("\n🧪 Testing single Lambda invocation...")
    
    try:
        test_payload = {
            "Records": [{
                "body": json.dumps({"informality_rate": 0.1, "seed": 999})
            }]
        }
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(test_payload)
        )
        
        result = json.loads(response['Payload'].read())
        print(f"Test invocation status: {response['StatusCode']}")
        print(f"Response: {result}")
        
        if 'errorMessage' in result:
            print(f"❌ Error in test: {result['errorMessage']}")
        else:
            print("✅ Test invocation successful")
            
    except Exception as e:
        print(f"❌ Error testing invocation: {str(e)}")

def check_sqs_dlq():
    """Check for dead letter queue messages"""
    print("\n💀 Checking for failed messages...")
    
    try:
        # List queues to find any DLQ
        queues = sqs.list_queues(QueueNamePrefix="ABM")
        print(f"Found queues: {queues.get('QueueUrls', [])}")
        
    except Exception as e:
        print(f"❌ Error checking SQS: {str(e)}")

def main():
    print("🚨 LAMBDA DIAGNOSTIC REPORT")
    print("=" * 50)
    
    check_lambda_errors()
    check_dynamodb_table()
    check_lambda_permissions()
    test_single_invocation()
    check_sqs_dlq()
    
    print("\n💡 TROUBLESHOOTING STEPS:")
    print("1. Check Lambda logs in AWS Console")
    print("2. Verify DynamoDB permissions")
    print("3. Test Lambda function manually")
    print("4. Check for import errors in Lambda")

if __name__ == "__main__":
    main()