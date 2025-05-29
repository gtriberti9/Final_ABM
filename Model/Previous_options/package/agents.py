
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
