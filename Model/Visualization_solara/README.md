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

This will execute:
- **Single Scenario Simulations**: Detailed analysis of low and high informality scenarios
- **Parameter Sweep Analysis**: Systematic exploration across different informality rates
- **Comparative Visualizations**: Plots showing relationships between informality and policy effectiveness
- **Statistical Analysis**: Summary statistics and convergence metrics

### Key Output Features
- **Macroeconomic Time Series**: Inflation, policy rates, and output gaps over time
- **Banking Sector Analysis**: Credit flows and interest rate differentials
- **Agent Distribution Analysis**: Wealth, pricing, and credit access patterns
- **Parameter Sensitivity**: How outcomes vary across informality levels

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
- **Initial Inflation Shock**: Starting inflation rate (8.81% - Peru post-COVID peak)
- **Simulation Length**: Maximum time steps (200 periods)
- **Convergence Criteria**: Inflation within ±1pp of target for 36 consecutive periods

## Dashboard Features

### Analysis Capabilities
The `app.py` script provides comprehensive analysis tools:
- **Scenario Comparison**: Direct comparison between low and high informality economies
- **Parameter Sensitivity Analysis**: Systematic exploration of key parameter relationships
- **Convergence Analysis**: Time-to-target metrics and policy effectiveness measures
- **Statistical Validation**: Robustness checks and confidence intervals

### Visualization Output
- **Macroeconomic Indicators**: Inflation, policy rate, output gap trajectories
- **Banking Sector Dynamics**: Credit flows, interest rate spreads by sector  
- **Agent Distributions**: Wealth, prices, and credit access by formality status
- **Parameter Sweep Results**: Relationships between informality and policy outcomes

### Data Export
- Simulation results automatically saved for further analysis
- Publication-ready plots generated in high resolution
- Statistical summaries exported for reporting

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

## Expected Results

Based on our research findings:
- **U-shaped Relationship**: Moderate informality (≈30%) may not severely impair policy effectiveness
- **Extended Adjustment**: High informality economies require longer convergence times
- **Credit Adaptation**: Banks adjust portfolios to serve informal markets as informality increases
- **Non-linear Effects**: Policy effectiveness varies non-monotonically with informality levels

## Extending the Model

Potential enhancements for future research:
- **Multi-market Equilibrium**: Endogenous labor and production markets
- **Institutional Heterogeneity**: Different regulatory frameworks across countries
- **Dynamic Informality**: Allow informality rates to respond to policy changes
- **Spatial Components**: Urban/rural or regional heterogeneity
- **Financial Sector Details**: Explicit modeling of bank balance sheets and capital requirements

## Technical Notes

### Performance Optimization
- Vectorized operations using NumPy for computational efficiency
- Configurable agent populations to balance realism and speed
- Parallel processing capabilities for parameter sweeps

### Reproducibility
- Fixed random seeds for consistent results
- Parameter logging for experiment replication
- Version control integration for model tracking

## Troubleshooting

### Common Issues
- **Slow Performance**: Reduce agent population in `model.py` or simulation length
- **Import Errors**: Ensure all required packages are installed (`pip install mesa numpy pandas matplotlib scipy`)
- **Memory Issues**: Close other applications or reduce parameter sweep ranges
- **Plot Display Issues**: Check matplotlib backend configuration

### Modifying Parameters
To adjust model parameters, edit the relevant values in:
- `agents.py`: Agent-specific parameters (risk premiums, behavioral rules)
- `model.py`: Model-wide parameters (agent populations, simulation length)
- `app.py`: Analysis parameters (informality rates to test, number of runs)

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

## License

This project is developed for academic research purposes. Please contact the author for usage permissions and collaboration opportunities.

---

**Author**: Giuliana Triberti  
**Institution**: University of Chicago - Masters in Computational Social Sciences  
**GitHub**: https://github.com/gtriberti9/ABM/tree/main/Midterm_2
