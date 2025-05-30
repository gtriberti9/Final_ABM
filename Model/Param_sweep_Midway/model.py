import numpy as np
import random
from typing import Dict, List

MAX_HISTORY = 500

class MonetaryPolicyModel:
    """
    Mesa-free Agent-Based Model for Monetary Policy with Central Bank, Commercial Banks, Firms, and Consumers
    Enhanced with Informality Dynamics
    """

    def __init__(self, 
                 n_firms=50, 
                 n_consumers=200, 
                 n_commercial_banks=5,
                 inflation_target=0.02,
                 initial_policy_rate=0.03,
                 informality_rate=0.1,
                 current_inflation=0.12):

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

        # Create agents (now as simple objects, not Mesa agents)
        self._create_agents()
        
        # Initialize price level based on initial firm prices
        self._update_price_level()
        self.prev_price_level = 10.0 * (1 + self.inflation_target) # Start with slight price growth expectation
        
        # Initialize informality statistics
        self._calculate_informality_stats()

        # Data collection (simplified without Mesa DataCollector)
        self.data_history = []

    def _create_agents(self):
        """Create all agent types as simple objects"""
        
        # Central Bank (singleton)
        self.central_bank = CentralBank(self)

        # Commercial Banks
        self.commercial_banks = []
        for i in range(self.n_commercial_banks):
            bank = CommercialBank(self, bank_id=i)
            self.commercial_banks.append(bank)

        # Firms
        self.firms = []
        for i in range(self.n_firms):
            firm = Firm(self, firm_id=i, informality_rate=self.informality_rate)
            self.firms.append(firm)

        # Consumers
        self.consumers = []
        for i in range(self.n_consumers):
            consumer = Consumer(self, consumer_id=i, informality_rate=self.informality_rate)
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

        # DYNAMIC potential that adjusts over time
        if hasattr(self, 'adaptive_potential'):
            # Gradually adjust potential toward actual utilization
            self.adaptive_potential = 0.95 * self.adaptive_potential + 0.05 * avg_utilization
        else:
            self.adaptive_potential = 0.8  # Initial value
            
        return (avg_utilization - self.adaptive_potential) / self.adaptive_potential

    def apply_taylor_rule(self):
        """Apply Taylor Rule to determine target interest rate"""
        inflation_gap = self.current_inflation - self.inflation_target
        target = (self.natural_rate + 
                 self.taylor_alpha * inflation_gap + 
                 self.taylor_beta * self.output_gap)

        return max(0.001, min(0.15, target))  # Bound between 0.1% and 15%

    def collect_data(self):
        """Collect current step data (replaces Mesa DataCollector)"""
        data = {
            "step": self.time,
            "Inflation": self.current_inflation,
            "Policy_Rate": self.policy_rate,
            "Output_Gap": self.output_gap,
            "Avg_Lending_Rate": np.mean([bank.lending_rate for bank in self.commercial_banks]),
            "Avg_Firm_Price": np.mean([firm.price for firm in self.firms]),
            "Total_Production": sum([firm.production for firm in self.firms]),
            "Total_Consumption": sum([consumer.consumption for consumer in self.consumers]),
            "Informal_Firm_Pct": len([f for f in self.firms if f.is_informal]) / len(self.firms) * 100,
            "Informal_Consumer_Pct": len([c for c in self.consumers if c.is_informal]) / len(self.consumers) * 100,
            "Credit_Access_Gap": self._calculate_credit_gap(),
            "Formal_Production": sum([f.production for f in self.firms if not f.is_informal]),
            "Informal_Production": sum([f.production for f in self.firms if f.is_informal]),
            "Total_Formal_Loans": sum([bank.formal_loans for bank in self.commercial_banks]),
            "Total_Informal_Loans": sum([bank.informal_loans for bank in self.commercial_banks])
        }
        self.data_history.append(data)

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
        self.collect_data()

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


# Simple agent classes (no Mesa inheritance)
class CentralBank:
    """Central Bank Agent - Sets monetary policy"""
    
    def __init__(self, model):
        self.model = model
        self.type = "central_bank"
        
    def step(self):
        """Central Bank actions - already handled in model.step()"""
        # Policy decisions are made at model level
        # This agent mainly exists for conceptual completeness
        pass


class CommercialBank:
    """Commercial Bank Agent - Intermediates between central bank and other agents"""
    
    def __init__(self, model, bank_id):
        self.model = model
        self.bank_id = bank_id
        self.type = "commercial_bank"
        
        # Bank parameters
        self.lending_rate = model.policy_rate + 0.02  # Spread over policy rate
        self.deposit_rate = model.policy_rate - 0.01  # Below policy rate
        self.capital_ratio = 0.12
        self.loan_portfolio = 0.0
        self.deposits = random.uniform(50, 200)  # Initial deposits
        self.max_lending_capacity = self.deposits * 10  # Leverage
        
        # Credit supply conditions
        self.credit_tightness = 0.5  # 0 = very loose, 1 = very tight
        
        # Track formal vs informal lending
        self.formal_loans = 0.0
        self.informal_loans = 0.0
        
        # Informality bias - banks prefer formal sector
        self.formal_sector_bias = random.uniform(0.6, 0.9)  # Higher = more bias toward formal
        
    def step(self):
        """Update rates and process loan applications"""
        # Update rates based on policy rate
        spread = random.uniform(0.015, 0.025)  # Variable spread
        self.lending_rate = self.model.policy_rate + spread
        self.deposit_rate = max(0.001, self.model.policy_rate - 0.005)
        
        # Update credit conditions based on economic environment
        if self.model.current_inflation > self.model.inflation_target * 1.5:
            self.credit_tightness = min(1.0, self.credit_tightness + 0.05)
        elif self.model.current_inflation < self.model.inflation_target * 0.5:
            self.credit_tightness = max(0.1, self.credit_tightness - 0.05)
    
    def process_loan_application(self, amount, applicant_type, is_informal=False):
        """Process loan application from firms or consumers"""
        # Check lending capacity
        if self.loan_portfolio + amount > self.max_lending_capacity:
            return False, None
        
        # Base credit scoring
        approval_probability = 0.8
        if applicant_type == "firm":
            approval_probability = 0.9 - self.credit_tightness * 0.3
        else:  # consumer
            approval_probability = 0.7 - self.credit_tightness * 0.4
        
        # Apply informality penalty
        if is_informal:
            # Informal agents face significant barriers
            informality_penalty = 0.3 + (1 - self.formal_sector_bias) * 0.2
            approval_probability *= (1 - informality_penalty)
            
            # Higher interest rate for informal sector
            effective_rate = self.lending_rate * (1 + 0.5 * self.formal_sector_bias)
        else:
            effective_rate = self.lending_rate
        
        if random.random() < approval_probability:
            self.loan_portfolio += amount
            if is_informal:
                self.informal_loans += amount
            else:
                self.formal_loans += amount
            return True, effective_rate
        
        return False, None


class Firm:
    """Firm Agent - Produces goods, sets prices, makes investment decisions"""
    
    def __init__(self, model, firm_id, informality_rate=0.1):
        self.model = model
        self.firm_id = firm_id
        self.type = "firm"
        
        # Determine if firm is informal (30% chance)
        self.is_informal = random.random() < informality_rate
        self.loan_interest_rate = None
        
        # Firm characteristics
        if self.is_informal:
            # Informal firms are typically smaller and less capitalized
            self.capacity = random.uniform(5, 25)
            self.cash = random.uniform(10, 50)
        else:
            self.capacity = random.uniform(10, 50)
            self.cash = random.uniform(20, 100)
            
        self.production = self.capacity * 0.8
        self.capacity_utilization = 0.8
        self.price = random.uniform(8, 12)
        self.prev_price = self.price
        
        # Costs and expectations
        if self.is_informal:
            # Informal firms often have higher costs due to inefficiencies
            self.marginal_cost = random.uniform(7, 9)
        else:
            self.marginal_cost = random.uniform(6, 8)
            
        self.inflation_expectation = model.current_inflation + random.uniform(-0.005, 0.005)
        self.markup = random.uniform(1.25, 1.4)  # Variable markup
        
        # Financial variables
        self.debt = 0.0
        self.credit_need = 0.0
        
        # Credit access score (0-1, higher = better access)
        if self.is_informal:
            self.credit_access_score = random.uniform(0.1, 0.4)
        else:
            self.credit_access_score = random.uniform(0.6, 0.9)
        
        # Price stickiness parameter
        self.price_stickiness = random.uniform(0.3, 0.7)  # Higher = more sticky
        
    def step(self):
        """Firm decision-making process"""
        # Update inflation expectations (adaptive with some forward-looking)
        self.update_inflation_expectations()
        
        # Update credit access based on performance
        self.update_credit_access()
        
        # Determine credit needs
        self.assess_credit_needs()
        
        # Apply for credit if needed
        if self.credit_need > 0:
            self.apply_for_credit()
        
        # Adjust production capacity if credit approved
        self.adjust_capacity()
        
        # Set prices
        self.set_price()
        
        # Produce goods
        self.produce()

        # Adjust production based on interest rate
        if self.loan_interest_rate is not None:
            rate_effect = max(0.1, 1 - 8 * (self.loan_interest_rate - 0.02))
        else:
            rate_effect = 1.0
        self.production = self.capacity * self.capacity_utilization * rate_effect

        # Adjust price more if utilization is high
        if self.capacity_utilization > 0.9:
            self.price *= 1.02
        elif self.capacity_utilization < 0.7:
            self.price *= 0.98
    
    def update_inflation_expectations(self):
        """Update inflation expectations using adaptive learning"""
        weight = 0.1  # Reduced from 0.15 for more gradual adjustment
        if len(self.model.inflation_history) > 1:
            recent_inflation = self.model.inflation_history[-1]
            noise = random.uniform(-0.003, 0.003)  # Reduced noise
            self.inflation_expectation = (weight * (recent_inflation + noise) +
                                        (1 - weight) * self.inflation_expectation)
    
    def update_credit_access(self):
        """Update credit access score based on performance and formality"""
        # Informal firms struggle to improve credit access
        if self.is_informal:
            # Slow improvement for informal firms, capped at 0.5
            if self.capacity_utilization > 0.8 and self.cash > 20:
                self.credit_access_score = min(0.5, self.credit_access_score + 0.01)
            elif self.cash < 10:
                self.credit_access_score = max(0.05, self.credit_access_score - 0.02)
        else:
            # Formal firms can improve more easily
            if self.capacity_utilization > 0.8 and self.cash > 30:
                self.credit_access_score = min(0.95, self.credit_access_score + 0.02)
            elif self.cash < 20:
                self.credit_access_score = max(0.3, self.credit_access_score - 0.01)
    
    def assess_credit_needs(self):
        """Determine if firm needs credit for expansion"""
        # Need credit if capacity utilization is high and cash is low
        threshold = 0.75 if self.is_informal else 0.85
        cash_threshold = 20 if self.is_informal else 30
        
        if self.capacity_utilization > threshold and self.cash < cash_threshold:
            if self.is_informal:
                self.credit_need = random.uniform(5, 20)  # Smaller amounts
            else:
                self.credit_need = random.uniform(10, 30)
        else:
            self.credit_need = 0.0
    
    def apply_for_credit(self):
        """Apply for credit from a commercial bank"""
        bank = self.model.get_random_bank()
        approved, rate = bank.process_loan_application(self.credit_need, "firm", self.is_informal)
        if approved:
            self.cash += self.credit_need
            self.debt += self.credit_need
            self.loan_interest_rate = rate  # Store the rate for future use
            self.credit_need = 0.0
    
    def adjust_capacity(self):
        # Get current lending rate
        avg_lending_rate = np.mean([bank.lending_rate for bank in self.model.commercial_banks])
        
        # MUCH stronger rate sensitivity
        base_threshold = 40 if self.is_informal else 50
        rate_multiplier = max(0.2, (avg_lending_rate / 0.05))  # Low rates = much easier expansion
        expansion_threshold = base_threshold * rate_multiplier
        
        utilization_threshold = 0.75 if self.is_informal else 0.8  # Lower threshold
        
        if self.cash > expansion_threshold and self.capacity_utilization > utilization_threshold:
            # BIGGER expansion when rates are low
            max_expansion = (15 if self.is_informal else 25) / rate_multiplier
            expansion = min(self.cash * 0.3, max_expansion)  # More aggressive
            self.capacity += expansion
            expansion_cost = 3 if self.is_informal else 2
            self.cash -= expansion * expansion_cost
    
    def set_price(self):
        """Set price using improved price-setting mechanism"""
        # Base price from markup over marginal cost
        base_price = self.marginal_cost * self.markup
        
        # Inflation expectations adjustment (more gradual)
        inflation_adj = 1 + (self.inflation_expectation * 1.2)  # Reduced impact
        
        # Capacity utilization adjustment (supply/demand balance)
        if self.capacity_utilization > 0.9:
            capacity_adj = 1.05  # Raise prices when near capacity
        elif self.capacity_utilization < 0.6:
            capacity_adj = 0.98  # Lower prices when underutilized
        else:
            capacity_adj = 1.0
        
        # Small random shock
        shock = random.uniform(0.98, 1.02)
        
        # Calculate target price
        target_price = base_price * inflation_adj * capacity_adj * shock
        
        # Apply price stickiness (Calvo-style pricing)
        if random.random() < (1 - self.price_stickiness * 0.5):
            # Firm adjusts price (only some firms adjust each period)
            adjustment_speed = 0.3
            new_price = (1 - adjustment_speed) * self.price + adjustment_speed * target_price
        else:
            # Price remains sticky
            new_price = self.price
        
        # Ensure minimum profitability
        min_price = self.marginal_cost * 1.05
        self.price = max(min_price, new_price)
        
        # Update marginal cost gradually based on input cost pressures
        cost_adjustment = 1 + (self.inflation_expectation * 0.3)
        self.marginal_cost *= (0.95 * 1 + 0.05 * cost_adjustment)  # Very gradual cost adjustment
    
    def produce(self):
        """Produce goods based on capacity and demand"""
        # Production affected by capacity utilization and random factors
        efficiency = random.uniform(0.88, 1.0) if self.is_informal else random.uniform(0.92, 1.0)
        self.production = self.capacity * self.capacity_utilization * efficiency
        
        # Update capacity utilization based on demand and economic conditions
        # Consider aggregate demand from consumers
        total_consumption = sum([c.consumption for c in self.model.consumers])
        total_capacity = sum([f.capacity for f in self.model.firms])
        
        if total_capacity > 0:
            demand_pressure = total_consumption / total_capacity
        else:
            demand_pressure = 0.8
            
        # Target utilization based on demand and price competitiveness
        avg_price = np.mean([f.price for f in self.model.firms])
        price_competitiveness = avg_price / self.price if self.price > 0 else 1.0
        
        base_utilization = min(0.95, demand_pressure * price_competitiveness)
        target_utilization = base_utilization * random.uniform(0.9, 1.1)
        
        # Gradual adjustment with bounds
        adjustment_speed = 0.15
        self.capacity_utilization += (target_utilization - self.capacity_utilization) * adjustment_speed
        self.capacity_utilization = max(0.3, min(0.98, self.capacity_utilization))


class Consumer:
    """Consumer Agent - Makes consumption and saving decisions"""
    
    def __init__(self, model, consumer_id, informality_rate=0.1):
        self.model = model
        self.consumer_id = consumer_id
        self.type = "consumer"
        
        # Determine if consumer works in informal sector (40% chance)
        self.is_informal = random.random() < informality_rate
        
        # Consumer characteristics
        if self.is_informal:
            # Informal workers typically earn less and have less wealth
            self.income = random.uniform(25, 60)
            self.wealth = random.uniform(5, 50)
        else:
            self.income = random.uniform(40, 80)
            self.wealth = random.uniform(20, 100)
            
        self.consumption = self.income * 0.8
        self.savings = self.wealth * 0.1
        
        # Expectations and preferences
        self.inflation_expectation = model.current_inflation + random.uniform(-0.01, 0.01)
        self.time_preference = random.uniform(0.02, 0.08)  # Discount rate
        self.risk_aversion = random.uniform(0.3, 0.7)
        
        # Financial behavior
        self.debt = random.uniform(0, 15) if self.is_informal else random.uniform(0, 20)
        self.propensity_to_borrow = random.uniform(0.05, 0.25) if self.is_informal else random.uniform(0.1, 0.4)
        
        # Credit access score (0-1, higher = better access)
        if self.is_informal:
            self.credit_access_score = random.uniform(0.05, 0.3)
        else:
            self.credit_access_score = random.uniform(0.5, 1)
        
    def step(self):
        """Consumer decision-making process"""
        # Update inflation expectations
        self.update_inflation_expectations()
        
        # Update credit access
        self.update_credit_access()
        
        # Make saving and borrowing decisions
        self.make_financial_decisions()
        
        # Adjust consumption
        self.adjust_consumption()
        
        # Purchase goods (implicit - affects firm demand)
        self.purchase_goods()
    
    def update_inflation_expectations(self):
        """Update inflation expectations using adaptive learning"""
        # Similar to firms but with more noise
        weight = 0.2
        if len(self.model.inflation_history) > 1:
            recent_inflation = self.model.inflation_history[-1]
            noise = random.uniform(-0.005, 0.005)  # Consumer perception noise
            self.inflation_expectation = (weight * (recent_inflation + noise) + 
                                        (1 - weight) * self.inflation_expectation)
    
    def update_credit_access(self):
        """Update credit access score based on income and formality"""
        if self.is_informal:
            # Very slow improvement for informal workers
            if self.income > 45 and self.wealth > 30:
                self.credit_access_score = min(0.35, self.credit_access_score + 0.005)
            elif self.wealth < 10:
                self.credit_access_score = max(0.01, self.credit_access_score - 0.01)
        else:
            # Formal workers can improve credit access
            if self.income > 60 and self.wealth > 50:
                self.credit_access_score = min(0.9, self.credit_access_score + 0.01)
            elif self.wealth < 15:
                self.credit_access_score = max(0.2, self.credit_access_score - 0.005)
    
    def make_financial_decisions(self):
        """Make saving and borrowing decisions based on interest rates"""
        avg_deposit_rate = np.mean([bank.deposit_rate for bank in self.model.commercial_banks])
        avg_lending_rate = np.mean([bank.lending_rate for bank in self.model.commercial_banks])
        
        # Real interest rate considerations
        real_deposit_rate = avg_deposit_rate - self.inflation_expectation
        real_lending_rate = avg_lending_rate - self.inflation_expectation
        
        # Saving decision
        if real_deposit_rate > self.time_preference:
            # Increase savings
            savings_rate = 0.08 if self.is_informal else 0.15
            additional_savings = min(self.income * savings_rate, self.consumption * 0.05)
            self.savings += additional_savings
            self.consumption -= additional_savings
        
        # Borrowing decision - much harder for informal workers
        max_debt_ratio = 0.3 if self.is_informal else 0.5
        if (real_lending_rate < self.time_preference * 2 and 
            self.debt < self.income * max_debt_ratio and 
            random.random() < self.propensity_to_borrow * self.credit_access_score):
            
            bank = self.model.get_random_bank()
            loan_amount = random.uniform(3, 10) if self.is_informal else random.uniform(5, 15)
            if bank and bank.process_loan_application(loan_amount, "consumer", self.is_informal):
                self.wealth += loan_amount
                self.debt += loan_amount
    
    def adjust_consumption(self):
        """Adjust consumption based on wealth, income, and expectations"""
        # Base consumption from permanent income
        consumption_rate = 0.8 if self.is_informal else 0.7
        base_consumption = self.income * consumption_rate
        
        # Wealth effect - smaller for informal workers due to precautionary savings
        wealth_multiplier = 0.03 if self.is_informal else 0.05
        wealth_effect = self.wealth * wealth_multiplier
        
        # Interest rate effect (substitution between present and future consumption)
        avg_real_rate = (np.mean([bank.deposit_rate for bank in self.model.commercial_banks]) - 
                        self.inflation_expectation)

        interest_sensitivity = 0.7 if self.is_informal else 1.0  # Increase sensitivity
        interest_effect = -avg_real_rate * self.income * interest_sensitivity
        
        # Uncertainty effect - stronger for informal workers
        uncertainty_multiplier = 1.5 if self.is_informal else 1.0
        uncertainty_effect = -abs(self.inflation_expectation - self.model.inflation_target) * self.income * uncertainty_multiplier
        
        self.consumption = (base_consumption + wealth_effect + 
                          interest_effect + uncertainty_effect)
        
        # Ensure consumption is reasonable
        min_consumption = self.income * 0.3 if self.is_informal else self.income * 0.4
        max_consumption = self.income * 1.1 if self.is_informal else self.income * 1.2
        self.consumption = max(min_consumption, min(max_consumption, self.consumption))
    
    def purchase_goods(self):
        """Purchase goods from firms (affects aggregate demand)"""
        # This is implicit in the model - consumption affects firm demand
        # In a more detailed model, this would involve actual transactions
        pass