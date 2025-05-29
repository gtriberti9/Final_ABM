# Inflation Targeting ABM with Banking Inclusion Effects

This Agent-Based Model (ABM) simulates how banking inclusion affects inflation targeting in developing economies. The model features households that can be either formal (banked) or informal (unbanked), firms operating in formal and informal sectors, and a central bank using interest rates to target inflation.

## Model Structure

The model is organized into three core files:

- **agents.py**: Defines the agent classes (Household, Firm, CentralBank)
- **model.py**: Implements the main model logic and environment
- **app.py**: Provides traditional visualization and data analysis tools

Plus an interactive dashboard:

- **dashboard.py**: Implements a Solara-based interactive dashboard

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Model

### Option 1: Traditional Visualization

To run the model with traditional matplotlib-based visualization:

```bash
python app.py
```

This will:
1. Run a basic simulation and show the results
2. Run comparative experiments with different banking inclusion rates
3. Display plots comparing the results

### Option 2: Interactive Dashboard (Recommended)

To run the model with the interactive Solara dashboard:

```bash
python app.py --dashboard
```

This will launch a web-based dashboard where you can:
- Adjust all model parameters interactively
- Run simulations and view results in real-time
- Compare different banking inclusion scenarios
- Analyze how quickly inflation returns to target

## Key Features

- **Banking Inclusion Effects**: The model captures how financial inclusion affects monetary policy transmission
  - Formal (banked) households respond more strongly to interest rate changes
  - Formal households have more accurate inflation expectations

- **Formal vs. Informal Sectors**: The economy is divided between formal and informal sectors
  - Formal firms are more responsive to central bank signals
  - Informal firms have less predictable pricing behavior

- **Inflation Targeting Mechanism**: The central bank uses a Taylor-type rule to adjust interest rates

## Model Parameters

- **Banking Inclusion Rate**: Proportion of households with access to banking services
- **Formal Sector Size**: Proportion of firms operating in the formal economy
- **Inflation Target**: Central bank's target inflation rate
- **Initial Inflation**: Starting inflation rate (to simulate a shock)
- **Inflation Shock Size**: Additional inflation shock applied at the start

## Extending the Model

This model provides a foundation for exploring inflation dynamics in developing economies. Some possible extensions:

- Add fiscal policy effects
- Include exchange rate dynamics
- Model financial sector explicitly
- Implement heterogeneous central bank credibility
- Add spatial components to represent urban/rural divides
