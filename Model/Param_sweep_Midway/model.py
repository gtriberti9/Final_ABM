#!/usr/bin/env python3
"""
Simplified Agent-Based Model for Monetary Policy
A cleaner, more readable version for intermediate coders

This model simulates:
- Central bank setting interest rates
- Commercial banks lending money
- Firms producing goods and setting prices
- Consumers buying goods and borrowing money
- Formal vs informal economic sectors
"""

import numpy as np
import random
from typing import Dict, List, Optional

# Global constants
MAX_HISTORY_LENGTH = 500  # How many past values to keep


class MonetaryPolicyModel:
    """
    Main model class that coordinates all agents and economic dynamics
    """
    
    def __init__(self, 
                 n_firms: int = 50,
                 n_consumers: int = 200,
                 n_banks: int = 5,
                 inflation_target: float = 0.02,  # 2% target
                 initial_rate: float = 0.03,     # 3% starting rate
                 informality_rate: float = 0.1): # 10% informal sector
        
        # Store basic parameters
        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_banks = n_banks
        self.inflation_target = inflation_target
        self.informality_rate = informality_rate
        
        # Economic indicators
        self.policy_rate = initial_rate
        self.current_inflation = 0.12  # Start with high inflation (12%)
        self.output_gap = 0.0
        self.price_level = 10.0  # Starting price level
        self.prev_price_level = 10.0
        
        # Taylor Rule parameters (how central bank responds)
        self.inflation_sensitivity = 1.5  # How much to raise rates when inflation high
        self.output_sensitivity = 0.5     # How much to respond to economic activity
        self.natural_rate = 0.025         # "Normal" interest rate
        
        # History tracking (for analysis)
        self.inflation_history = [self.current_inflation]
        self.policy_rate_history = [self.policy_rate]
        self.output_gap_history = [self.output_gap]
        
        # Time counter
        self.time_step = 0
        
        # Create all agents
        self._create_agents()
        
        # Data collection for analysis
        self.collected_data = []
    
    def _create_agents(self):
        """Create all the agents in our economy"""
        
        # Create one central bank
        self.central_bank = CentralBank(self)
        
        # Create commercial banks
        self.banks = []
        for i in range(self.n_banks):
            bank = CommercialBank(self, bank_id=i)
            self.banks.append(bank)
        
        # Create firms (some formal, some informal)
        self.firms = []
        for i in range(self.n_firms):
            # Randomly decide if firm is informal
            is_informal = random.random() < self.informality_rate
            firm = Firm(self, firm_id=i, is_informal=is_informal)
            self.firms.append(firm)
        
        # Create consumers (some formal, some informal)
        self.consumers = []
        for i in range(self.n_consumers):
            # Randomly decide if consumer works in informal sector
            is_informal = random.random() < self.informality_rate
            consumer = Consumer(self, consumer_id=i, is_informal=is_informal)
            self.consumers.append(consumer)
    
    def calculate_inflation(self) -> float:
        """
        Calculate current inflation rate based on price changes
        """
        if self.time_step == 0:
            return self.current_inflation
        
        # Update current price level based on firm prices
        total_production = sum(firm.production for firm in self.firms)
        if total_production > 0:
            # Weighted average price by production
            self.price_level = sum(firm.price * firm.production for firm in self.firms) / total_production
        
        # Calculate inflation as percentage change in prices
        if self.prev_price_level > 0:
            raw_inflation = (self.price_level - self.prev_price_level) / self.prev_price_level
        else:
            raw_inflation = 0.0
        
        # Add some persistence (inflation doesn't change instantly)
        persistence = 0.1
        noise = random.uniform(-0.002, 0.002)  # Small random component
        
        new_inflation = (persistence * self.current_inflation + 
                        (1 - persistence) * (raw_inflation + self.inflation_target * 0.6) + 
                        noise)
        
        # Keep inflation in reasonable bounds
        return max(-0.05, min(0.15, new_inflation))  # Between -5% and 15%
    
    def calculate_output_gap(self) -> float:
        """
        Calculate how much the economy is above/below its potential
        Positive = economy running hot, negative = economy slack
        """
        if not self.firms:
            return 0.0
        
        # Calculate average capacity utilization
        avg_utilization = np.mean([firm.capacity_utilization for firm in self.firms])
        
        # Potential output slowly adjusts to actual output
        if hasattr(self, 'potential_output'):
            self.potential_output = 0.95 * self.potential_output + 0.05 * avg_utilization
        else:
            self.potential_output = 0.8  # Initial assumption
        
        # Output gap = (actual - potential) / potential
        return (avg_utilization - self.potential_output) / self.potential_output
    
    def apply_taylor_rule(self) -> float:
        """
        Central bank's rule for setting interest rates
        Rate = natural_rate + inflation_response + output_response
        """
        inflation_gap = self.current_inflation - self.inflation_target
        
        target_rate = (self.natural_rate + 
                      self.inflation_sensitivity * inflation_gap + 
                      self.output_sensitivity * self.output_gap)
        
        # Keep rate between 0.1% and 15%
        return max(0.001, min(0.15, target_rate))
    
    def step(self):
        """
        Run one time period of the simulation
        This is the main function that coordinates everything
        """
        
        # Store previous price level for inflation calculation
        self.prev_price_level = self.price_level
        
        # === PHASE 1: Central Bank Actions ===
        # Calculate current economic conditions
        self.current_inflation = self.calculate_inflation()
        self.output_gap = self.calculate_output_gap()
        
        # Set new policy rate based on Taylor rule
        target_rate = self.apply_taylor_rule()
        
        # Gradually adjust policy rate (central banks don't change rates drastically)
        adjustment_speed = 0.2
        rate_change = (target_rate - self.policy_rate) * adjustment_speed
        self.policy_rate += rate_change
        self.policy_rate = max(0.001, min(0.15, self.policy_rate))
        
        # Central bank updates its stance
        self.central_bank.step()
        
        # === PHASE 2: Commercial Banks ===
        # Banks adjust their rates based on central bank rate
        for bank in self.banks:
            bank.step()
        
        # === PHASE 3: Firms ===
        # Firms make production and pricing decisions
        for firm in self.firms:
            firm.step()
        
        # === PHASE 4: Consumers ===
        # Consumers make consumption and borrowing decisions
        for consumer in self.consumers:
            consumer.step()
        
        # === PHASE 5: End of Period Updates ===
        # Update history
        self.inflation_history.append(self.current_inflation)
        self.policy_rate_history.append(self.policy_rate)
        self.output_gap_history.append(self.output_gap)
        
        # Keep history from getting too long
        if len(self.inflation_history) > MAX_HISTORY_LENGTH:
            self.inflation_history = self.inflation_history[-MAX_HISTORY_LENGTH:]
            self.policy_rate_history = self.policy_rate_history[-MAX_HISTORY_LENGTH:]
            self.output_gap_history = self.output_gap_history[-MAX_HISTORY_LENGTH:]
        
        # Collect data for analysis
        self.collect_data()
        
        # Increment time
        self.time_step += 1
    
    def collect_data(self):
        """Collect current state data for later analysis"""
        
        # Calculate some summary statistics
        avg_firm_price = np.mean([firm.price for firm in self.firms])
        total_production = sum(firm.production for firm in self.firms)
        total_consumption = sum(consumer.consumption for consumer in self.consumers)
        avg_lending_rate = np.mean([bank.lending_rate for bank in self.banks])
        
        # Count formal vs informal
        informal_firms = len([f for f in self.firms if f.is_informal])
        informal_consumers = len([c for c in self.consumers if c.is_informal])
        
        # Store data point
        data_point = {
            "step": self.time_step,
            "inflation": self.current_inflation,
            "policy_rate": self.policy_rate,
            "output_gap": self.output_gap,
            "avg_lending_rate": avg_lending_rate,
            "avg_firm_price": avg_firm_price,
            "total_production": total_production,
            "total_consumption": total_consumption,
            "informal_firms_pct": (informal_firms / self.n_firms) * 100,
            "informal_consumers_pct": (informal_consumers / self.n_consumers) * 100
        }
        
        self.collected_data.append(data_point)
    
    def get_random_bank(self) -> Optional['CommercialBank']:
        """Get a random bank for transactions"""
        return random.choice(self.banks) if self.banks else None
    
    def is_converged(self, tolerance: float = 0.005, periods: int = 10) -> bool:
        """Check if inflation has converged close to target"""
        if len(self.inflation_history) < periods:
            return False
        
        recent_inflation = self.inflation_history[-periods:]
        avg_recent = np.mean(recent_inflation)
        
        return abs(avg_recent - self.inflation_target) < tolerance


class CentralBank:
    """
    Central Bank Agent - Sets monetary policy
    In real life: Federal Reserve, European Central Bank, etc.
    """
    
    def __init__(self, model: MonetaryPolicyModel):
        self.model = model
    
    def step(self):
        """
        Central bank actions each period
        In this simple model, the main actions happen in the model's step() function
        """
        # In a more complex model, central bank might:
        # - Communicate policy changes
        # - Adjust reserve requirements
        # - Conduct quantitative easing
        pass


class CommercialBank:
    """
    Commercial Bank Agent - Lends money to firms and consumers
    Examples: Chase, Bank of America, Wells Fargo
    """
    
    def __init__(self, model: MonetaryPolicyModel, bank_id: int):
        self.model = model
        self.bank_id = bank_id
        
        # Bank's financial position
        self.deposits = random.uniform(50, 200)  # Customer deposits
        self.loans_outstanding = 0.0             # Total loans made
        self.max_leverage = 10                   # Can lend 10x deposits
        
        # Interest rates the bank charges/pays
        self.lending_rate = model.policy_rate + 0.02  # Charge 2% above policy rate
        self.deposit_rate = model.policy_rate - 0.01  # Pay 1% below policy rate
        
        # Credit policy
        self.credit_tightness = 0.5  # How strict lending is (0=loose, 1=tight)
        
        # Track lending to formal vs informal sectors
        self.formal_loans = 0.0
        self.informal_loans = 0.0
        
        # Banks prefer formal sector (less risky)
        self.formal_bias = random.uniform(0.6, 0.9)  # Higher = prefer formal sector
    
    def step(self):
        """Update bank's rates and lending conditions each period"""
        
        # Update lending rate based on central bank rate
        spread = random.uniform(0.015, 0.025)  # Variable profit margin
        self.lending_rate = self.model.policy_rate + spread
        self.deposit_rate = max(0.001, self.model.policy_rate - 0.005)
        
        # Adjust credit conditions based on economic environment
        if self.model.current_inflation > self.model.inflation_target * 1.5:
            # High inflation -> tighten credit
            self.credit_tightness = min(1.0, self.credit_tightness + 0.05)
        elif self.model.current_inflation < self.model.inflation_target * 0.5:
            # Low inflation -> loosen credit
            self.credit_tightness = max(0.1, self.credit_tightness - 0.05)
    
    def process_loan_request(self, amount: float, applicant_type: str, is_informal: bool) -> tuple[bool, Optional[float]]:
        """
        Decide whether to approve a loan application
        
        Returns: (approved: bool, interest_rate: float or None)
        """
        
        # Check if bank has capacity to lend
        max_lending = self.deposits * self.max_leverage
        if self.loans_outstanding + amount > max_lending:
            return False, None
        
        # Base approval probability
        if applicant_type == "firm":
            approval_chance = 0.9 - self.credit_tightness * 0.3
        else:  # consumer
            approval_chance = 0.7 - self.credit_tightness * 0.4
        
        # Penalty for informal sector (harder to evaluate, higher risk)
        if is_informal:
            informality_penalty = 0.3 + (1 - self.formal_bias) * 0.2
            approval_chance *= (1 - informality_penalty)
            # Higher interest rate for informal sector
            effective_rate = self.lending_rate * (1 + 0.5 * self.formal_bias)
        else:
            effective_rate = self.lending_rate
        
        # Make lending decision
        if random.random() < approval_chance:
            # Approve loan
            self.loans_outstanding += amount
            if is_informal:
                self.informal_loans += amount
            else:
                self.formal_loans += amount
            return True, effective_rate
        else:
            # Reject loan
            return False, None


class Firm:
    """
    Firm Agent - Produces goods and sets prices
    Can be formal (registered, pays taxes) or informal (unregistered)
    """
    
    def __init__(self, model: MonetaryPolicyModel, firm_id: int, is_informal: bool):
        self.model = model
        self.firm_id = firm_id
        self.is_informal = is_informal
        
        # Production capacity and efficiency
        if is_informal:
            # Informal firms are typically smaller
            self.capacity = random.uniform(5, 25)
            self.cash = random.uniform(10, 50)
        else:
            self.capacity = random.uniform(10, 50)
            self.cash = random.uniform(20, 100)
        
        # Production and pricing
        self.production = self.capacity * 0.8
        self.capacity_utilization = 0.8  # Using 80% of capacity
        self.price = random.uniform(8, 12)
        
        # Costs
        if is_informal:
            # Informal firms often have higher costs (inefficiencies, bribes, etc.)
            self.marginal_cost = random.uniform(7, 9)
        else:
            self.marginal_cost = random.uniform(6, 8)
        
        # Financial variables
        self.debt = 0.0
        self.markup = random.uniform(1.25, 1.4)  # Price = cost * markup
        
        # Expectations about future inflation
        self.inflation_expectation = model.current_inflation + random.uniform(-0.005, 0.005)
        
        # Access to formal credit system
        if is_informal:
            self.credit_access = random.uniform(0.1, 0.4)  # Poor access
        else:
            self.credit_access = random.uniform(0.6, 0.9)  # Good access
        
        # How sticky are prices (reluctance to change prices)
        self.price_stickiness = random.uniform(0.3, 0.7)
    
    def step(self):
        """Firm's decision-making each period"""
        
        # 1. Update expectations about inflation
        self.update_inflation_expectations()
        
        # 2. Check if need credit for expansion
        if self.should_seek_credit():
            self.apply_for_credit()
        
        # 3. Adjust production capacity if possible
        self.adjust_capacity()
        
        # 4. Set prices for this period
        self.set_price()
        
        # 5. Produce goods
        self.produce()
    
    def update_inflation_expectations(self):
        """Update beliefs about future inflation based on recent experience"""
        learning_rate = 0.1
        if len(self.model.inflation_history) > 1:
            recent_inflation = self.model.inflation_history[-1]
            noise = random.uniform(-0.003, 0.003)
            
            # Adaptive learning: slowly update expectations
            self.inflation_expectation = (learning_rate * (recent_inflation + noise) +
                                        (1 - learning_rate) * self.inflation_expectation)
    
    def should_seek_credit(self) -> bool:
        """Decide if firm needs to borrow money"""
        # Need credit if:
        # 1. Running at high capacity (need to expand)
        # 2. Low on cash
        capacity_threshold = 0.75 if self.is_informal else 0.85
        cash_threshold = 20 if self.is_informal else 30
        
        return (self.capacity_utilization > capacity_threshold and 
                self.cash < cash_threshold)
    
    def apply_for_credit(self):
        """Try to get a loan from a bank"""
        bank = self.model.get_random_bank()
        if not bank:
            return
        
        # How much to borrow
        if self.is_informal:
            loan_amount = random.uniform(5, 20)  # Smaller loans
        else:
            loan_amount = random.uniform(10, 30)
        
        # Apply for loan
        approved, interest_rate = bank.process_loan_request(loan_amount, "firm", self.is_informal)
        
        if approved:
            self.cash += loan_amount
            self.debt += loan_amount
    
    def adjust_capacity(self):
        """Expand production capacity if profitable and affordable"""
        
        # Consider current interest rates
        avg_lending_rate = np.mean([bank.lending_rate for bank in self.model.banks])
        
        # Higher rates make expansion more expensive
        expansion_threshold = (40 if self.is_informal else 50) * (avg_lending_rate / 0.05)
        utilization_threshold = 0.75 if self.is_informal else 0.8
        
        if (self.cash > expansion_threshold and 
            self.capacity_utilization > utilization_threshold):
            
            # Expand capacity
            max_expansion = 15 if self.is_informal else 25
            expansion = min(self.cash * 0.3, max_expansion)
            
            self.capacity += expansion
            expansion_cost = 3 if self.is_informal else 2  # Cost per unit capacity
            self.cash -= expansion * expansion_cost
    
    def set_price(self):
        """Set price for goods based on costs and expectations"""
        
        # Base price from markup over marginal cost
        base_price = self.marginal_cost * self.markup
        
        # Adjust for expected inflation
        inflation_adjustment = 1 + (self.inflation_expectation * 1.2)
        
        # Adjust based on capacity utilization (supply/demand balance)
        if self.capacity_utilization > 0.9:
            capacity_adjustment = 1.05  # Raise prices when demand high
        elif self.capacity_utilization < 0.6:
            capacity_adjustment = 0.98  # Lower prices when demand low
        else:
            capacity_adjustment = 1.0
        
        # Small random factor
        random_shock = random.uniform(0.98, 1.02)
        
        # Calculate target price
        target_price = base_price * inflation_adjustment * capacity_adjustment * random_shock
        
        # Apply price stickiness (don't change prices too often)
        if random.random() < (1 - self.price_stickiness * 0.5):
            # Adjust price gradually
            adjustment_speed = 0.3
            new_price = (1 - adjustment_speed) * self.price + adjustment_speed * target_price
        else:
            # Keep price unchanged
            new_price = self.price
        
        # Ensure minimum profitability
        min_price = self.marginal_cost * 1.05
        self.price = max(min_price, new_price)
        
        # Update marginal cost gradually
        cost_pressure = 1 + (self.inflation_expectation * 0.3)
        self.marginal_cost *= (0.95 + 0.05 * cost_pressure)
    
    def produce(self):
        """Produce goods based on capacity and demand conditions"""
        
        # Production efficiency
        efficiency = random.uniform(0.88, 1.0) if self.is_informal else random.uniform(0.92, 1.0)
        
        # Base production
        self.production = self.capacity * self.capacity_utilization * efficiency
        
        # Adjust capacity utilization based on market conditions
        # Consider overall demand in economy
        total_consumption = sum(consumer.consumption for consumer in self.model.consumers)
        total_capacity = sum(firm.capacity for firm in self.model.firms)
        
        if total_capacity > 0:
            demand_pressure = total_consumption / total_capacity
        else:
            demand_pressure = 0.8
        
        # Consider price competitiveness
        avg_price = np.mean([firm.price for firm in self.model.firms])
        price_competitiveness = avg_price / self.price if self.price > 0 else 1.0
        
        # Target capacity utilization
        target_utilization = min(0.95, demand_pressure * price_competitiveness)
        target_utilization *= random.uniform(0.9, 1.1)  # Add some randomness
        
        # Gradually adjust capacity utilization
        adjustment_speed = 0.15
        self.capacity_utilization += (target_utilization - self.capacity_utilization) * adjustment_speed
        self.capacity_utilization = max(0.3, min(0.98, self.capacity_utilization))


class Consumer:
    """
    Consumer Agent - Buys goods and services, may borrow money
    Can work in formal sector (regular job) or informal sector (gig work, cash economy)
    """
    
    def __init__(self, model: MonetaryPolicyModel, consumer_id: int, is_informal: bool):
        self.model = model
        self.consumer_id = consumer_id
        self.is_informal = is_informal
        
        # Income and wealth
        if is_informal:
            # Informal workers typically earn less
            self.income = random.uniform(25, 60)
            self.wealth = random.uniform(5, 50)
        else:
            self.income = random.uniform(40, 80)
            self.wealth = random.uniform(20, 100)
        
        # Spending and saving
        self.consumption = self.income * 0.8  # Spend 80% of income
        self.savings = self.wealth * 0.1
        
        # Financial behavior
        self.debt = random.uniform(0, 15) if is_informal else random.uniform(0, 20)
        self.risk_aversion = random.uniform(0.3, 0.7)  # How risk-averse
        
        # Expectations and preferences
        self.inflation_expectation = model.current_inflation + random.uniform(-0.01, 0.01)
        self.time_preference = random.uniform(0.02, 0.08)  # How impatient
        
        # Access to credit
        if is_informal:
            self.credit_access = random.uniform(0.05, 0.3)  # Very limited access
        else:
            self.credit_access = random.uniform(0.5, 0.9)   # Good access
        
        # Borrowing behavior
        self.borrowing_propensity = random.uniform(0.05, 0.25) if is_informal else random.uniform(0.1, 0.4)
    
    def step(self):
        """Consumer's decision-making each period"""
        
        # 1. Update inflation expectations
        self.update_inflation_expectations()
        
        # 2. Make financial decisions (saving, borrowing)
        self.make_financial_decisions()
        
        # 3. Decide how much to consume
        self.adjust_consumption()
        
        # 4. Update wealth and debt
        self.update_financial_position()
    
    def update_inflation_expectations(self):
        """Update beliefs about future inflation"""
        learning_rate = 0.2  # Consumers learn faster than firms
        if len(self.model.inflation_history) > 1:
            recent_inflation = self.model.inflation_history[-1]
            noise = random.uniform(-0.005, 0.005)  # More noise in consumer perceptions
            
            self.inflation_expectation = (learning_rate * (recent_inflation + noise) +
                                        (1 - learning_rate) * self.inflation_expectation)
    
    def make_financial_decisions(self):
        """Decide whether to save more or borrow"""
        
        avg_deposit_rate = np.mean([bank.deposit_rate for bank in self.model.banks])
        avg_lending_rate = np.mean([bank.lending_rate for bank in self.model.banks])
        
        # Real interest rates (adjusted for inflation expectations)
        real_deposit_rate = avg_deposit_rate - self.inflation_expectation
        real_lending_rate = avg_lending_rate - self.inflation_expectation
        
        # Saving decision
        if real_deposit_rate > self.time_preference:
            # Good returns on saving -> save more
            savings_rate = 0.08 if self.is_informal else 0.15
            additional_savings = min(self.income * savings_rate, self.consumption * 0.05)
            self.savings += additional_savings
            self.consumption -= additional_savings
        
        # Borrowing decision
        max_debt_ratio = 0.3 if self.is_informal else 0.5  # Max debt relative to income
        
        should_borrow = (real_lending_rate < self.time_preference * 2 and  # Cheap credit
                        self.debt < self.income * max_debt_ratio and      # Not too indebted
                        random.random() < self.borrowing_propensity * self.credit_access)  # Willing and able
        
        if should_borrow:
            bank = self.model.get_random_bank()
            if bank:
                loan_amount = random.uniform(3, 10) if self.is_informal else random.uniform(5, 15)
                approved, rate = bank.process_loan_request(loan_amount, "consumer", self.is_informal)
                
                if approved:
                    self.wealth += loan_amount
                    self.debt += loan_amount
    
    def adjust_consumption(self):
        """Decide how much to spend this period"""
        
        # Base consumption from income
        consumption_rate = 0.8 if self.is_informal else 0.7
        base_consumption = self.income * consumption_rate
        
        # Wealth effect (rich people spend more)
        wealth_multiplier = 0.03 if self.is_informal else 0.05
        wealth_effect = self.wealth * wealth_multiplier
        
        # Interest rate effect
        avg_real_rate = (np.mean([bank.deposit_rate for bank in self.model.banks]) - 
                        self.inflation_expectation)
        
        # Higher interest rates -> save more, consume less
        interest_sensitivity = 0.7 if self.is_informal else 1.0
        interest_effect = -avg_real_rate * self.income * interest_sensitivity
        
        # Uncertainty effect (when inflation uncertain, people save more)
        uncertainty_multiplier = 1.5 if self.is_informal else 1.0
        uncertainty = abs(self.inflation_expectation - self.model.inflation_target)
        uncertainty_effect = -uncertainty * self.income * uncertainty_multiplier
        
        # Total consumption
        self.consumption = (base_consumption + wealth_effect + 
                          interest_effect + uncertainty_effect)
        
        # Keep consumption reasonable
        min_consumption = self.income * 0.3 if self.is_informal else self.income * 0.4
        max_consumption = self.income * 1.1 if self.is_informal else self.income * 1.2
        self.consumption = max(min_consumption, min(max_consumption, self.consumption))
    
    def update_financial_position(self):
        """Update wealth and debt over time"""
        # In a more complete model, would track:
        # - Interest payments on debt
        # - Interest earned on savings
        # - Asset price changes
        pass


# Example usage and testing
if __name__ == "__main__":
    print("Testing Simple ABM Model...")
    
    # Create a model with 20 firms, 100 consumers, 3 banks
    # 30% informality rate
    model = MonetaryPolicyModel(
        n_firms=20,
        n_consumers=100,
        n_banks=3,
        informality_rate=0.3
    )
    
    print(f"Created model with:")
    print(f"  {model.n_firms} firms ({len([f for f in model.firms if f.is_informal])} informal)")
    print(f"  {model.n_consumers} consumers ({len([c for c in model.consumers if c.is_informal])} informal)")
    print(f"  {model.n_banks} banks")
    print(f"  Starting inflation: {model.current_inflation:.1%}")
    print(f"  Target inflation: {model.inflation_target:.1%}")
    print(f"  Starting policy rate: {model.policy_rate:.1%}")
    
    # Run simulation for 50 steps
    print("\nRunning simulation...")
    for step in range(50):
        model.step()
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: Inflation={model.current_inflation:.2%}, "
                  f"Policy Rate={model.policy_rate:.2%}, "
                  f"Output Gap={model.output_gap:.3f}")
    
    # Final results
    print(f"\nFinal Results (after {model.time_step} steps):")
    print(f"  Final inflation: {model.current_inflation:.2%}")
    print(f"  Final policy rate: {model.policy_rate:.2%}")
    print(f"  Converged to target: {model.is_converged()}")
    
    # Show some economic statistics
    total_production = sum(firm.production for firm in model.firms)
    total_consumption = sum(consumer.consumption for consumer in model.consumers)
    avg_firm_price = np.mean([firm.price for firm in model.firms])
    
    print(f"  Total production: {total_production:.1f}")
    print(f"  Total consumption: {total_consumption:.1f}")
    print(f"  Average price: {avg_firm_price:.2f}")
    
    print("\nModel test completed successfully!")