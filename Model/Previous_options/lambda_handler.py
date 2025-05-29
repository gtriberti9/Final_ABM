import boto3
import json
from decimal import Decimal
from model import MonetaryPolicyModel  # Make sure your model code is zipped with the Lambda

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ABMResults')

def run_simulation(informality_rate, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    model = MonetaryPolicyModel(
        n_firms=50,
        n_consumers=200,
        n_commercial_banks=5,
        inflation_target=0.02,
        initial_policy_rate=0.03,
        informality_rate=informality_rate,
        current_inflation=0.12
    )
    max_steps = 200
    for step in range(max_steps):
        model.step()
        if len(model.inflation_history) >= 12 and all(
            0.018 <= x <= 0.022 for x in model.inflation_history[-12:]
        ):
            break
    inflation = model.inflation_history
    inflation_target = model.inflation_target
    policy_rate_changes = sum(
        abs(model.policy_rate_history[i] - model.policy_rate_history[i-1]) > 1e-6
        for i in range(1, len(model.policy_rate_history))
    )
    inflation_history_str = [str(x) for x in inflation]
    result = {
        "informality_rate": Decimal(str(informality_rate)),
        "seed": int(seed),
        "time_to_target": int(len(inflation)),
        "inflation_volatility": Decimal(str(np.std(inflation))),
        "max_deviation": Decimal(str(max(abs(x - inflation_target) for x in inflation))),
        "policy_rate_changes": int(policy_rate_changes),
        "final_inflation": Decimal(str(inflation[-1])),
        "inflation_history": inflation_history_str
    }
    return result

def lambda_handler(event, context):
    for record in event.get('Records', []):
        params = json.loads(record['body'])
        result = run_simulation(params['informality_rate'], params['seed'])
        table.put_item(Item=result)
    return {'statusCode': 200}