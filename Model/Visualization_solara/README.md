# Inflation Targeting ABM - Visualization and Analysis

This folder contains the visualization and analysis tools for the Inflation Targeting Agent-Based Model (ABM) that explores how informality and banking inclusion affect monetary policy effectiveness in developing economies.

## Project Overview

This Agent-Based Model simulates how different levels of economic informality impact central banks' ability to target inflation. The model features:
- **Heterogeneous Agents**: Formal and informal households, firms, and commercial banks
- **Central Bank**: Uses Taylor Rule for inflation targeting
- **Credit Markets**: Differential access to credit between formal and informal sectors
- **Comprehensive Analysis**: Parameter sweeps and comparative scenario analysis

## Folder Structure

```
Visualization_solara/
├── __pycache__/      # Python cache files
├── agents.py         # Agent class definitions (Household, Firm, CentralBank, CommercialBank)
├── model.py          # Main ABM model logic and simulation environment
├── app.py           # Visualization runner and analysis tools
└── README.md        # This file
```

## Core Model Components

### Agent Classes (`agents.py`)
- **Household**: Formal/informal consumers with different credit access and inflation expectations
- **Firm**: Formal/informal producers with heterogeneous pricing and credit constraints  
- **CentralBank**: Implements Taylor Rule for interest rate setting
- **CommercialBank**: Intermediates between central bank and agents with sector-specific risk premiums

### Model Environment (`model.py`)
- Coordinates agent interactions and market clearing
- Implements monetary policy transmission mechanisms
- Tracks macroeconomic indicators (inflation, output gap, credit flows)
- Handles parameter sweeps for sensitivity analysis

## Installation

1. Navigate to the `Visualization_solara` folder:
```bash
cd Visualization_solara
```

2. Install required dependencies only if they are not in the environment:
```bash
pip install mesa=3.1.4 numpy pandas matplotlib scipy
```

Required packages include:
- `mesa` - Agent-based modeling framework
- `numpy`, `pandas` - Data manipulation and analysis
- `matplotlib` - Visualization and plotting
- `scipy` - Statistical analysis

## Running the Model

### Main Analysis Tool
Run the complete analysis with visualization:
```bash
solara run app.py
```

## Dashboard Features

### Visualization Output
- **Macroeconomic Indicators**: Inflation, policy rate, output gap trajectories
- **Banking Sector Dynamics**: Credit flows, interest rate spreads by sector  
- **Agent Distributions**: Wealth, prices, and credit access by formality status
- **Informality Indicators**: Figures of nformality and credit gap

### Parameter sweep

1. Navigate to the `Param_sweep_AWS` folder:
```bash
cd Param_sweep_AWS
```

2. Run:
```bash
python main_runner.py --quick-test
python simple_analysis.py
```

## Key Model Parameters

### Economic Structure
- **Informality Rate**: Proportion of agents operating in informal sector (0-100%)
- **Formal Sector Productivity**: Relative efficiency of formal vs informal firms
- **Credit Access Gap**: Differential in lending rates between sectors

### Monetary Policy
- **Inflation Target**: Central bank's target rate (default: 2%)
- **Taylor Rule Coefficients**: 
  - α (inflation gap weight): 1.5
  - β (output gap weight): 0.5
- **Natural Interest Rate**: Long-term equilibrium rate (6.42% for Peru)

### Initial Conditions
- **Initial Inflation Shock**: Starting inflation rate (12% - Peru post-COVID peak)
- **Simulation Length**: Maximum time steps (200 periods)


## Key Research Questions

The model addresses:
1. **Main Question**: Are economies with higher informal markets prone to lagged inflation targeting?
2. **Policy Transmission**: How do different informality levels affect monetary policy effectiveness?
3. **Credit Market Dynamics**: How do banks adapt their lending to serve informal sectors?
4. **Convergence Patterns**: What factors determine time-to-target for inflation?

## Model Validation

The model is calibrated using Peruvian economic data:
- Informality rates from ILO/World Bank data (71.66% for Peru)
- Central bank parameters from Banco Central de Reserva del Perú
- Post-pandemic inflation shock magnitude (8.81% peak)

### Reproducibility
- Fixed random seeds for consistent results
- Parameter logging for experiment replication
- Version control integration for model tracking

### Performance Tips
- Start with smaller agent populations for testing
- Use shorter simulation periods during development
- Monitor memory usage during parameter sweeps

## Citation

If using this model for research, please cite:
```
Triberti, G. (2025). Exploring the Role of Agent-Based Models in Inflation Targeting Post-crisis. 
Masters in Computational Social Sciences, University of Chicago.
```

---

**Author**: Giuliana Triberti  
**Institution**: University of Chicago - Masters in Computational Social Sciences  
**GitHub**: https://github.com/gtriberti9/Final_ABM/edit/main/Model/Visualization_solara
