import solara
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import asyncio
import time
from Model.Param_sweep_AWS.ABM_Model.model import MonetaryPolicyModel

# Global variables to store model state
model = None
is_running = solara.reactive(False)
current_step = solara.reactive(0)
max_steps = solara.reactive(200)

# Model parameters
n_firms = solara.reactive(50)
n_consumers = solara.reactive(200)
n_banks = solara.reactive(5)
inflation_target = solara.reactive(0.02)
initial_policy_rate = solara.reactive(0.03)
informality_rate = solara.reactive(0.1)
current_inflation = solara.reactive(0.02)  

# Add trigger for UI updates
ui_trigger = solara.reactive(0)

def reset_model():
    """Reset the model with current parameters"""
    global model
    model = MonetaryPolicyModel(
        n_firms=n_firms.value,
        n_consumers=n_consumers.value,
        n_commercial_banks=n_banks.value,
        inflation_target=inflation_target.value,
        initial_policy_rate=initial_policy_rate.value,
        informality_rate = informality_rate.value,
        current_inflation=current_inflation.value
    )
    current_step.set(0)
    is_running.set(False)
    ui_trigger.set(ui_trigger.value + 1)  # Force UI update

def step_model():
    """Execute one step of the model"""
    global model
    if model is not None:
        model.step()
        current_step.set(current_step.value + 1)
        ui_trigger.set(ui_trigger.value + 1)  # Force UI update

def inflation_stable():
    if len(model.inflation_history) < 36:
        return False
    last_12 = model.inflation_history[-36:]
    return all(0.01 <= val <= 0.03 for val in last_12)

async def run_model_async():
    """Run the model asynchronously with real-time updates"""
    global model
    if model is None:
        return
        
    is_running.set(True)
    
    while current_step.value < max_steps.value and is_running.value:
        step_model()
        if inflation_stable():
            break
        await asyncio.sleep(0.1)  # Small delay for visualization
        
    is_running.set(False)

def run_model():
    """Start async model run"""
    if model is not None and current_step.value < max_steps.value:
        asyncio.create_task(run_model_async())

def stop_model():
    """Stop the running model"""
    is_running.set(False)

@solara.component
def ModelControls():
    """Control panel for the ABM"""
    # Use ui_trigger to force re-render
    _ = ui_trigger.value
    
    with solara.Card("Model Controls", margin=10):
        with solara.Row():
            with solara.Column():
                with solara.Div(style={"width": "300px"}):
                    solara.SliderInt("Firms", value=n_firms, min=10, max=100)
                with solara.Div(style={"width": "300px"}):
                    solara.SliderInt("Consumers", value=n_consumers, min=50, max=500)
                with solara.Div(style={"width": "300px"}):
                    solara.SliderInt("Banks", value=n_banks, min=2, max=10)
                with solara.Div(style={"width": "300px"}):
                    solara.SliderFloat("Informality rate", value=informality_rate, min=0.1, max=0.8, step=0.01)
            with solara.Column():
                with solara.Div(style={"width": "300px"}):
                    solara.SliderFloat("Inflation Target", value=inflation_target, min=0.01, max=0.05, step=0.005)
                with solara.Div(style={"width": "300px"}):
                    solara.SliderFloat("Initial Policy Rate", value=initial_policy_rate, min=0.0, max=0.1, step=0.005)
                with solara.Div(style={"width": "300px"}):
                    solara.SliderInt("Max Steps", value=max_steps, min=50, max=500)
                with solara.Div(style={"width": "300px"}):
                    solara.SliderFloat("Inflation Shock", value=current_inflation, min=0.0, max=0.2, step=0.01)
        with solara.Row():
            solara.Button("Reset Model", on_click=lambda: reset_model(), disabled=is_running.value)
            solara.Button("Step", on_click=lambda: step_model(), disabled=is_running.value or model is None)
            if is_running.value:
                solara.Button("Stop", on_click=lambda: stop_model(), color="error")
            else:
                solara.Button("Run", on_click=lambda: run_model(), disabled=model is None, color="success")
        
        if model is not None:
            solara.Markdown(f"**Current Step:** {current_step.value}")
            solara.Markdown(f"**Current Inflation:** {model.current_inflation:.3f} ({model.current_inflation*100:.1f}%)")
            solara.Markdown(f"**Policy Rate:** {model.policy_rate:.3f} ({model.policy_rate*100:.1f}%)")
            solara.Markdown(f"**Output Gap:** {model.output_gap:.3f}")
            
            # Add informality statistics
            if hasattr(model, 'informal_stats'):
                stats = model.informal_stats
                solara.Markdown(f"**Informal Firms:** {stats['informal_firms']} ({stats['informal_firm_pct']:.1f}%)")
                solara.Markdown(f"**Informal Consumers:** {stats['informal_consumers']} ({stats['informal_consumer_pct']:.1f}%)")
                solara.Markdown(f"**Credit Access Gap:** {stats['credit_gap']:.2f}")

@solara.component
def MacroIndicators():
    """Display main macroeconomic indicators with matplotlib"""
    # Use ui_trigger to force re-render
    _ = ui_trigger.value
    
    if model is None or len(model.inflation_history) < 2:
        solara.Info("Run the model to see macro indicators")
        return

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    steps = np.arange(len(model.inflation_history))
    ax.plot(steps, np.array(model.inflation_history)*100, label="Inflation (%)", color="red", linewidth=2)
    ax.plot(steps, np.array(model.policy_rate_history)*100, label="Policy Rate (%)", color="blue", linewidth=2)
    ax.plot(steps, np.array(model.output_gap_history)*100, label="Output Gap (%)", color="green", linewidth=2)
    ax.axhline(model.inflation_target*100, color="red", linestyle="--", alpha=0.5, label="Inflation Target")
    
    ax.set_xlim(0, len(model.inflation_history) - 1)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Percent (%)")
    ax.set_title("Macroeconomic Indicators Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    solara.FigureMatplotlib(fig)

@solara.component
def AgentAnalytics():
    """Display agent analytics including informality"""
    # Use ui_trigger to force re-render
    _ = ui_trigger.value
    
    if model is None:
        solara.Info("Run the model to see agent analytics")
        return

    fig = Figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Firm prices by formality
    formal_firms = [f for f in model.firms if not f.is_informal]
    informal_firms = [f for f in model.firms if f.is_informal]
    
    if formal_firms:
        ax1.hist([f.price for f in formal_firms], bins=15, alpha=0.7, label="Formal", color="blue")
    if informal_firms:
        ax1.hist([f.price for f in informal_firms], bins=15, alpha=0.7, label="Informal", color="red")
    ax1.set_title("Firm Prices by Sector")
    ax1.set_xlabel("Price")
    ax1.legend()
    
    # Consumer wealth by formality
    formal_consumers = [c for c in model.consumers if not c.is_informal]
    informal_consumers = [c for c in model.consumers if c.is_informal]
    
    if formal_consumers:
        ax2.hist([c.wealth for c in formal_consumers], bins=15, alpha=0.7, label="Formal", color="blue")
    if informal_consumers:
        ax2.hist([c.wealth for c in informal_consumers], bins=15, alpha=0.7, label="Informal", color="red")
    ax2.set_title("Consumer Wealth by Sector")
    ax2.set_xlabel("Wealth")
    ax2.legend()
    
    # Credit access comparison
    formal_firm_credit = np.mean([f.credit_access_score for f in formal_firms]) if formal_firms else 0
    informal_firm_credit = np.mean([f.credit_access_score for f in informal_firms]) if informal_firms else 0
    formal_consumer_credit = np.mean([c.credit_access_score for c in formal_consumers]) if formal_consumers else 0
    informal_consumer_credit = np.mean([c.credit_access_score for c in informal_consumers]) if informal_consumers else 0
    
    categories = ['Formal\nFirms', 'Informal\nFirms', 'Formal\nConsumers', 'Informal\nConsumers']
    credit_scores = [formal_firm_credit, informal_firm_credit, formal_consumer_credit, informal_consumer_credit]
    colors = ['blue', 'red', 'blue', 'red']
    
    ax3.bar(categories, credit_scores, color=colors, alpha=0.7)
    ax3.set_title("Average Credit Access Score")
    ax3.set_ylabel("Credit Access (0-1)")
    ax3.set_ylim(0, 1)
    
    fig.tight_layout()
    solara.FigureMatplotlib(fig)

@solara.component
def BankingSystem():
    """Display banking system analytics with matplotlib"""
    # Use ui_trigger to force re-render
    _ = ui_trigger.value
    
    if model is None or not model.commercial_banks:
        solara.Info("Run the model to see banking analytics")
        return

    with solara.Card("Banking System"):
        bank_data = []
        for i, bank in enumerate(model.commercial_banks):
            bank_data.append({
                'Bank': f'Bank {i+1}',
                'Lending Rate': bank.lending_rate * 100,
                'Deposit Rate': bank.deposit_rate * 100,
                #'Loan Portfolio': bank.loan_portfolio,
                #'Credit Tightness': bank.credit_tightness,
                'Deposits': bank.deposits,
                'Formal Loans': bank.formal_loans,
                'Informal Loans': bank.informal_loans
            })
        df = pd.DataFrame(bank_data)

        # Matplotlib plots for banking system
        fig = Figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax4 = fig.add_subplot(122)

        # Interest rates
        width = 0.35
        x = np.arange(len(df))
        ax1.bar(x, df['Lending Rate'], width, label='Lending Rate', color='red', alpha=0.7)
        ax1.bar(x, df['Deposit Rate'], width, label='Deposit Rate', color='blue', alpha=0.7)
        ax1.set_title('Interest Rates (%)')
        ax1.set_ylabel('Rate (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Bank'], rotation=45)
        ax1.legend()

        
        # Formal vs Informal Loans
        width = 0.35
        ax4.bar(x - width/2, df['Formal Loans'], width, label='Formal Loans', color='blue', alpha=0.7)
        ax4.bar(x + width/2, df['Informal Loans'], width, label='Informal Loans', color='red', alpha=0.7)
        ax4.set_title('Loans by Sector')
        ax4.set_ylabel('Amount')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['Bank'], rotation=45)
        ax4.legend()

        fig.tight_layout()
        solara.FigureMatplotlib(fig)

        # Show data table
        solara.DataFrame(df)

@solara.component
def InformalityAnalysis():
    """New component to analyze informality dynamics"""
    # Use ui_trigger to force re-render
    _ = ui_trigger.value
    
    if model is None or len(model.informality_history) < 2:
        solara.Info("Run the model to see informality analysis")
        return
        
    with solara.Card("Informality Dynamics"):
        fig = Figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        steps = np.arange(len(model.informality_history))
        
        # Informality rates over time
        informal_firm_rates = [data['informal_firm_pct'] for data in model.informality_history]
        informal_consumer_rates = [data['informal_consumer_pct'] for data in model.informality_history]
        
        ax1.plot(steps, informal_firm_rates, label="Informal Firms (%)", color="red", linewidth=2)
        ax1.plot(steps, informal_consumer_rates, label="Informal Consumers (%)", color="orange", linewidth=2)
        ax1.set_ylabel("Informality Rate (%)")
        ax1.set_title("Informality Rates Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Credit gap over time
        credit_gaps = [data['credit_gap'] for data in model.informality_history]
        ax2.plot(steps, credit_gaps, label="Credit Access Gap", color="purple", linewidth=2)
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Credit Gap")
        ax2.set_title("Credit Access Gap (Formal - Informal)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        solara.FigureMatplotlib(fig)

@solara.component
def Page():
    """Main app layout"""
    with solara.Column():
        solara.Title("Agent-Based Model: Monetary Policy & Banking System with Informality")
        ModelControls()
        MacroIndicators()
        with solara.Row():
            AgentAnalytics()
            BankingSystem()
        InformalityAnalysis()

# Initialize the model on app start
reset_model()