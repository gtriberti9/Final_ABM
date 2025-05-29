
import boto3
import json
import random
import numpy as np
from decimal import Decimal
from model import MonetaryPolicyModel

def run_simulation(informality_rate, seed):
    """Run simplified simulation"""
    random.seed(seed)
    np.random.seed(seed)
    
    model = MonetaryPolicyModel(
        n_firms=20,  # Reduced for faster execution
        n_consumers=50,
        n_commercial_banks=3,
        inflation_target=0.02,
        initial_policy_rate=0.03,
        informality_rate=informality_rate,
        current_inflation=0.12
    )
    
    max_steps = 100  # Reduced for faster execution
    for step in range(max_steps):
        model.step()
        if len(model.inflation_history) >= 12 and all(
            0.018 <= x <= 0.022 for x in model.inflation_history[-12:]
        ):
            break
    
    # Calculate results
    inflation = model.inflation_history
    inflation_target = model.inflation_target
    
    result = {
        "informality_rate": Decimal(str(informality_rate)),
        "seed": int(seed),
        "time_to_target": int(len(inflation)),
        "inflation_volatility": Decimal(str(np.std(inflation))),
        "max_deviation": Decimal(str(max(abs(x - inflation_target) for x in inflation))),
        "final_inflation": Decimal(str(inflation[-1])),
        "steps_completed": int(model.time)
    }
    return result

def lambda_handler(event, context):
    """Lambda handler function"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ABMResults')
    
    try:
        for record in event.get('Records', []):
            params = json.loads(record['body'])
            result = run_simulation(params['informality_rate'], params['seed'])
            table.put_item(Item=result)
            
        return {'statusCode': 200, 'body': 'Success'}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'statusCode': 500, 'body': f'Error: {str(e)}'}
