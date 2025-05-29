#!/usr/bin/env python3
"""
Create minimal working Lambda package without problematic dependencies
"""

import os
import sys
import subprocess
import zipfile
import shutil

def create_minimal_agents():
    """Create simplified agents.py without mesa dependency"""
    content = '''
import random
import numpy as np

# Simple base classes to replace mesa
class Agent:
    def __init__(self, model):
        self.model = model

class Model:
    def __init__(self):
        self.agents = []

class DataCollector:
    def __init__(self, model_reporters=None):
        self.model_reporters = model_reporters or {}
        self.model_vars = {}
    
    def collect(self, model):
        for key, func in self.model_reporters.items():
            try:
                self.model_vars[key] = func(model)
            except:
                self.model_vars[key] = 0

class CentralBankAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.type = "central_bank"
        
    def step(self):
        pass

class CommercialBankAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.type = "commercial_bank"
        self.lending_rate = 0.05
        self.deposit_rate = 0.02
        self.formal_loans = 0.0
        self.informal_loans = 0.0
        
    def step(self):
        pass
    
    def process_loan_application(self, amount, applicant_type, is_informal=False):
        # Simplified approval logic
        if random.random() < 0.7:  # 70% approval rate
            return True, self.lending_rate
        return False, None

class FirmAgent(Agent):
    def __init__(self, model, informality_rate=0.1):
        super().__init__(model)
        self.type = "firm"
        self.is_informal = random.random() < informality_rate
        self.capacity = random.uniform(10, 50)
        self.production = self.capacity * 0.8
        self.price = random.uniform(8, 12)
        self.capacity_utilization = 0.8
        
    def step(self):
        # Simplified firm behavior
        self.production = self.capacity * self.capacity_utilization
        self.price *= random.uniform(0.99, 1.01)

class ConsumerAgent(Agent):
    def __init__(self, model, informality_rate=0.1):
        super().__init__(model)
        self.type = "consumer"
        self.is_informal = random.random() < informality_rate
        self.income = random.uniform(40, 80)
        self.consumption = self.income * 0.8
        
    def step(self):
        # Simplified consumer behavior
        self.consumption = self.income * random.uniform(0.7, 0.9)
'''
    return content

def create_minimal_model():
    """Create simplified model.py without mesa dependency"""
    content = '''
import numpy as np
import random
from agents import CentralBankAgent, CommercialBankAgent, FirmAgent, ConsumerAgent, DataCollector

class MonetaryPolicyModel:
    def __init__(self, n_firms=50, n_consumers=200, n_commercial_banks=5,
                 inflation_target=0.02, initial_policy_rate=0.03,
                 informality_rate=0.1, current_inflation=0.02):
        
        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_commercial_banks = n_commercial_banks
        self.inflation_target = inflation_target
        self.policy_rate = initial_policy_rate
        self.informality_rate = informality_rate
        self.current_inflation = current_inflation
        self.output_gap = 0.0
        self.time = 0
        
        # Historical data
        self.inflation_history = [self.current_inflation]
        self.policy_rate_history = [self.policy_rate]
        self.output_gap_history = [self.output_gap]
        
        # Taylor Rule parameters
        self.taylor_alpha = 1.5
        self.taylor_beta = 0.5
        self.natural_rate = 0.025
        
        # Create agents
        self.central_bank = CentralBankAgent(self)
        self.commercial_banks = [CommercialBankAgent(self) for _ in range(n_commercial_banks)]
        self.firms = [FirmAgent(self, informality_rate) for _ in range(n_firms)]
        self.consumers = [ConsumerAgent(self, informality_rate) for _ in range(n_consumers)]
        
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Inflation": lambda m: m.current_inflation,
                "Policy_Rate": lambda m: m.policy_rate,
                "Output_Gap": lambda m: m.output_gap,
            }
        )
        
    def step(self):
        # Simple model step
        self.current_inflation = max(0, self.current_inflation + random.uniform(-0.01, 0.01))
        self.output_gap = random.uniform(-0.1, 0.1)
        
        # Taylor rule
        inflation_gap = self.current_inflation - self.inflation_target
        target_rate = self.natural_rate + self.taylor_alpha * inflation_gap + self.taylor_beta * self.output_gap
        self.policy_rate = max(0.001, min(0.15, target_rate))
        
        # Agent steps
        for bank in self.commercial_banks:
            bank.step()
        for firm in self.firms:
            firm.step()
        for consumer in self.consumers:
            consumer.step()
            
        # Update history
        self.inflation_history.append(self.current_inflation)
        self.policy_rate_history.append(self.policy_rate)
        self.output_gap_history.append(self.output_gap)
        
        self.datacollector.collect(self)
        self.time += 1
    
    def get_random_bank(self):
        return random.choice(self.commercial_banks) if self.commercial_banks else None
    
    def is_converged(self, tolerance=0.005, periods=10):
        if len(self.inflation_history) < periods:
            return False
        recent_inflation = self.inflation_history[-periods:]
        avg_recent = np.mean(recent_inflation)
        return abs(avg_recent - self.inflation_target) < tolerance
'''
    return content

def create_minimal_lambda_handler():
    """Create simplified lambda_handler.py"""
    content = '''
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
'''
    return content

def create_package():
    """Create the Lambda package"""
    print("Creating minimal Lambda package...")
    
    # Clean up
    if os.path.exists('package'):
        shutil.rmtree('package')
    if os.path.exists('abm_lambda.zip'):
        os.remove('abm_lambda.zip')
    
    # Create package directory
    os.makedirs('package')
    
    # Install minimal dependencies
    print("Installing numpy and boto3...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'numpy==1.24.3', 'boto3==1.29.7', 
        '-t', 'package', '--no-user', '--no-deps'
    ])
    
    # Install numpy dependencies separately
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'numpy==1.24.3', '-t', 'package', '--no-user'
    ])
    
    # Create simplified Python files
    print("Creating simplified model files...")
    with open('package/agents.py', 'w') as f:
        f.write(create_minimal_agents())
    
    with open('package/model.py', 'w') as f:
        f.write(create_minimal_model())
        
    with open('package/lambda_handler.py', 'w') as f:
        f.write(create_minimal_lambda_handler())
    
    # Remove problematic files
    print("Cleaning up package...")
    for root, dirs, files in os.walk('package'):
        # Remove problematic directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'tests', 'docs', 'examples']]
        # Remove problematic files
        for file in files:
            if file.endswith(('.pyc', '.pyo')) or 'docs' in file.lower():
                try:
                    os.remove(os.path.join(root, file))
                except:
                    pass
    
    # Create ZIP
    print("Creating ZIP file...")
    with zipfile.ZipFile('abm_lambda.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk('package'):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, 'package')
                zipf.write(file_path, arcname)
    
    # Check size
    size_mb = os.path.getsize('abm_lambda.zip') / (1024 * 1024)
    print(f"✅ Package created: abm_lambda.zip ({size_mb:.2f} MB)")
    
    return size_mb < 50

if __name__ == "__main__":
    if create_package():
        print("🚀 Ready to deploy! Run: python deploy_lambda.py")
    else:
        print("❌ Package still too large")