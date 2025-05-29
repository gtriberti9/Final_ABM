import mesa
import numpy as np
from typing import Dict, List
import random

MAX_HISTORY = 500

class MonetaryPolicyModel(mesa.Model):
    """
    Agent-Based Model for Monetary Policy with Central Bank, Commercial Banks, Firms, and Consumers
    Enhanced with Informality Dynamics
    """

    def __init__(self, 
                 n_firms=50, 
                 n_consumers=200, 
                 n_commercial_banks=5,
                 inflation_target=0.02,
                 initial_policy_rate=0.03,
                 informality_rate=0.1,
                 current_inflation=0.02):

        super().__init__()

        # Model parameters
        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_commercial_banks = n_commercial_banks
        self.inflation_target = inflation_target
        self.policy_rate = initial_policy_rate
        self.informality_rate = informality_rate

        # Economic indicators
        self.current_inflation = current_inflation
        self.output_gap = 0.0
        self.target_rate = initial_policy_rate
        self.avg_capacity_utilization = 0.8

        # Taylor Rule parameters
        self.taylor_alpha = 1.5  # Response to inflation
        self.taylor_beta = 0.5  # Response to output gap
        self.natural_rate = 0.025

        # Historical data for tracking
        self.inflation_history = [self.current_inflation]
        self.policy_rate_history = [self.policy_rate]
        self.output_gap_history = [self.output_gap]
        
        # Store previous aggregate price level for inflation calculation
        self.prev_price_level = 10.0  # Initialize with reasonable price level
        self.price_level = 10.0
        
        # Informality tracking
        self.informality_history = []
        self.informal_stats = {}

        # Time variable (step counter)
        self.time = 0

        # Create agents
        self._create_agents()
        
        # Initialize price level based on initial firm prices
        self._update_price_level()
        self.prev_price_level = 10.0 * (1 + self.inflation_target) # Start with slight price growth expectation
        
        # Initialize informality statistics
        self._calculate_informality_stats()

        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Inflation": lambda m: m.current_inflation,
                "Policy_Rate": lambda m: m.policy_rate,
                "Output_Gap": lambda m: m.output_gap,
                "Avg_Lending_Rate": lambda m: np.mean([bank.lending_rate for bank in m.commercial_banks]),
                "Avg_Firm_Price": lambda m: np.mean([firm.price for firm in m.firms]),
                "Total_Production": lambda m: sum([firm.production for firm in m.firms]),
                "Total_Consumption": lambda m: sum([consumer.consumption for consumer in m.consumers]),
                "Informal_Firm_Pct": lambda m: len([f for f in m.firms if f.is_informal]) / len(m.firms) * 100,
                "Informal_Consumer_Pct": lambda m: len([c for c in m.consumers if c.is_informal]) / len(m.consumers) * 100,
                "Credit_Access_Gap": lambda m: m._calculate_credit_gap(),
                "Formal_Production": lambda m: sum([f.production for f in m.firms if not f.is_informal]),
                "Informal_Production": lambda m: sum([f.production for f in m.firms if f.is_informal]),
                "Total_Formal_Loans": lambda m: sum([bank.formal_loans for bank in m.commercial_banks]),
                "Total_Informal_Loans": lambda m: sum([bank.informal_loans for bank in m.commercial_banks])
            }
        )

    def _create_agents(self):
        """Create all agent types"""
        from agents import CentralBankAgent, CommercialBankAgent, FirmAgent, ConsumerAgent

        # Central Bank (singleton)
        self.central_bank = CentralBankAgent(self)

        # Commercial Banks
        self.commercial_banks = []
        for i in range(self.n_commercial_banks):
            bank = CommercialBankAgent(self)
            self.commercial_banks.append(bank)

        # Firms
        self.firms = []
        for i in range(self.n_firms):
            firm = FirmAgent(self, informality_rate=self.informality_rate)
            self.firms.append(firm)

        # Consumers
        self.consumers = []
        for i in range(self.n_consumers):
            consumer = ConsumerAgent(self, informality_rate=self.informality_rate)
            self.consumers.append(consumer)

    def _update_price_level(self):
        """Update aggregate price level based on firm prices weighted by production"""
        if not self.firms:
            return

        total_production = sum(firm.production for firm in self.firms)
        if total_production == 0:
            # Use capacity as weights if no production yet
            total_weight = sum(firm.capacity for firm in self.firms)
            if total_weight > 0:
                self.price_level = sum(firm.price * firm.capacity for firm in self.firms) / total_weight
        else:
            self.price_level = sum(firm.price * firm.production for firm in self.firms) / total_production

    def _calculate_informality_stats(self):
        """Calculate current informality statistics"""
        total_firms = len(self.firms)
        total_consumers = len(self.consumers)
        
        informal_firms = len([f for f in self.firms if f.is_informal])
        informal_consumers = len([c for c in self.consumers if c.is_informal])
        
        self.informal_stats = {
            'informal_firms': informal_firms,
            'informal_consumers': informal_consumers,
            'informal_firm_pct': (informal_firms / total_firms * 100) if total_firms > 0 else 0,
            'informal_consumer_pct': (informal_consumers / total_consumers * 100) if total_consumers > 0 else 0,
            'credit_gap': self._calculate_credit_gap()
        }
    
    def _calculate_credit_gap(self):
        """Calculate the credit access gap between formal and informal sectors"""
        formal_firms = [f for f in self.firms if not f.is_informal]
        informal_firms = [f for f in self.firms if f.is_informal]
        formal_consumers = [c for c in self.consumers if not c.is_informal]
        informal_consumers = [c for c in self.consumers if c.is_informal]
        
        # Average credit access scores
        formal_firm_access = np.mean([f.credit_access_score for f in formal_firms]) if formal_firms else 0
        informal_firm_access = np.mean([f.credit_access_score for f in informal_firms]) if informal_firms else 0
        formal_consumer_access = np.mean([c.credit_access_score for c in formal_consumers]) if formal_consumers else 0
        informal_consumer_access = np.mean([c.credit_access_score for c in informal_consumers]) if informal_consumers else 0
        
        # Overall gap (formal - informal)
        firm_gap = formal_firm_access - informal_firm_access
        consumer_gap = formal_consumer_access - informal_consumer_access
        
        return (firm_gap + consumer_gap) / 2

    def calculate_inflation_rate(self):
        """Calculate current inflation rate based on price level changes"""
        if self.time == 0:
            return self.current_inflation
            
        # Update current price level
        self._update_price_level()
        
        # Calculate inflation as percentage change in price level
        if self.prev_price_level > 0:
            inflation = (self.price_level - self.prev_price_level) / self.prev_price_level
        else:
            inflation = 0.0
        
        # Add some persistence and noise to make inflation more realistic
        persistence = 0.1
        noise = random.uniform(0.000, 0.004)
        
        # Combine calculated inflation with previous inflation (persistence) and noise
        new_inflation = persistence * self.current_inflation + (1 - persistence) * (inflation + self.inflation_target * 0.6) + noise
        
        # Bound inflation to reasonable range
        return max(-0.05, min(0.15, new_inflation))

    def calculate_output_gap(self):
        """Calculate output gap based on capacity utilization, accounting for informality"""
        if not self.firms:
            return 0.0

        # Separate formal and informal firms
        formal_firms = [f for f in self.firms if not f.is_informal]
        informal_firms = [f for f in self.firms if f.is_informal]
        
        # Calculate weighted average utilization
        if formal_firms and informal_firms:
            formal_weight = sum(f.capacity for f in formal_firms)
            informal_weight = sum(f.capacity for f in informal_firms)
            total_weight = formal_weight + informal_weight
            
            if total_weight > 0:
                formal_utilization = np.mean([f.capacity_utilization for f in formal_firms])
                informal_utilization = np.mean([f.capacity_utilization for f in informal_firms])
                
                avg_utilization = ((formal_utilization * formal_weight + 
                                  informal_utilization * informal_weight) / total_weight)
            else:
                avg_utilization = 0.8
        elif formal_firms:
            avg_utilization = np.mean([f.capacity_utilization for f in formal_firms])
        elif informal_firms:
            avg_utilization = np.mean([f.capacity_utilization for f in informal_firms])
        else:
            avg_utilization = 0.8

        self.avg_capacity_utilization = avg_utilization

        # Output gap: deviation from potential (formal sector potential = 0.85, informal = 0.75)
        formal_potential = 0.85
        informal_potential = 0.75
        
        # DYNAMIC potential that adjusts over time
        if hasattr(self, 'adaptive_potential'):
            # Gradually adjust potential toward actual utilization
            self.adaptive_potential = 0.95 * self.adaptive_potential + 0.05 * avg_utilization
        else:
            self.adaptive_potential = 0.8  # Initial value

        # Weighted potential based on sector composition
        # if formal_firms and informal_firms:
        #     formal_weight = len(formal_firms)
        #     informal_weight = len(informal_firms)
        #     total_agents = formal_weight + informal_weight
            
        #     weighted_potential = ((formal_potential * formal_weight + 
        #                          informal_potential * informal_weight) / total_agents)
        # elif formal_firms:
        #     weighted_potential = formal_potential
        # else:
        #     weighted_potential = informal_potential
            
        return (avg_utilization - self.adaptive_potential) / self.adaptive_potential

    def apply_taylor_rule(self):
        """Apply Taylor Rule to determine target interest rate"""
        inflation_gap = self.current_inflation - self.inflation_target
        target = (self.natural_rate + 
                 self.taylor_alpha * inflation_gap + 
                 self.taylor_beta * self.output_gap)

        return max(0.001, min(0.15, target))  # Bound between 0.1% and 15%

    def step(self):
        """Execute one step of the model following the 5-phase pipeline"""

        # Store previous price level before agents act
        self.prev_price_level = self.price_level

        # Phase 1: Central Bank Phase
        self.current_inflation = self.calculate_inflation_rate()
        self.output_gap = self.calculate_output_gap()
        self.target_rate = self.apply_taylor_rule()

        # Gradual adjustment of policy rate
        adjustment_speed = 0.2
        rate_change = (self.target_rate - self.policy_rate) * adjustment_speed
        self.policy_rate += rate_change
        self.policy_rate = max(0.001, min(0.15, self.policy_rate))

        # Central bank acts
        self.central_bank.step()

        # Phase 2: Commercial Bank Phase
        for bank in self.commercial_banks:
            bank.step()

        # Phase 3: Firm Phase
        for firm in self.firms:
            firm.step()

        # Phase 4: Consumer Phase
        for consumer in self.consumers:
            consumer.step()

        # Phase 5: End of Period
        self.inflation_history.append(self.current_inflation)
        self.policy_rate_history.append(self.policy_rate)
        self.output_gap_history.append(self.output_gap)
        
        # Update informality statistics
        self._calculate_informality_stats()
        self.informality_history.append(self.informal_stats.copy())

        # Keep history manageable
        if len(self.inflation_history) > MAX_HISTORY:
            self.inflation_history = self.inflation_history[-MAX_HISTORY:]
            self.policy_rate_history = self.policy_rate_history[-MAX_HISTORY:]
            self.output_gap_history = self.output_gap_history[-MAX_HISTORY:]
            self.informality_history = self.informality_history[-MAX_HISTORY:]

        # Collect data
        self.datacollector.collect(self)

        # Increment time step
        self.time += 1

    def get_random_bank(self):
        """Get a random commercial bank for transactions"""
        return random.choice(self.commercial_banks) if self.commercial_banks else None

    def is_converged(self, tolerance=0.005, periods=10):
        """Check if inflation has converged to target"""
        if len(self.inflation_history) < periods:
            return False

        recent_inflation = self.inflation_history[-periods:]
        avg_recent = np.mean(recent_inflation)

        return abs(avg_recent - self.inflation_target) < tolerance
    
    def get_sector_analysis(self):
        """Get detailed analysis of formal vs informal sectors"""
        formal_firms = [f for f in self.firms if not f.is_informal]
        informal_firms = [f for f in self.firms if f.is_informal]
        formal_consumers = [c for c in self.consumers if not c.is_informal]
        informal_consumers = [c for c in self.consumers if c.is_informal]
        
        analysis = {
            'formal_firms': {
                'count': len(formal_firms),
                'avg_production': np.mean([f.production for f in formal_firms]) if formal_firms else 0,
                'avg_price': np.mean([f.price for f in formal_firms]) if formal_firms else 0,
                'avg_credit_access': np.mean([f.credit_access_score for f in formal_firms]) if formal_firms else 0,
                'total_capacity': sum([f.capacity for f in formal_firms])
            },
            'informal_firms': {
                'count': len(informal_firms),
                'avg_production': np.mean([f.production for f in informal_firms]) if informal_firms else 0,
                'avg_price': np.mean([f.price for f in informal_firms]) if informal_firms else 0,
                'avg_credit_access': np.mean([f.credit_access_score for f in informal_firms]) if informal_firms else 0,
                'total_capacity': sum([f.capacity for f in informal_firms])
            },
            'formal_consumers': {
                'count': len(formal_consumers),
                'avg_consumption': np.mean([c.consumption for c in formal_consumers]) if formal_consumers else 0,
                'avg_credit_access': np.mean([c.credit_access_score for c in formal_consumers]) if formal_consumers else 0
            },
            'informal_consumers': {
                'count': len(informal_consumers),
                'avg_consumption': np.mean([c.consumption for c in informal_consumers]) if informal_consumers else 0,
                'avg_credit_access': np.mean([c.credit_access_score for c in informal_consumers]) if informal_consumers else 0
            }
        }
        return analysis