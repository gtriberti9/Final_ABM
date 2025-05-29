
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
