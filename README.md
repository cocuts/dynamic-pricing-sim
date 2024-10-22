# Dynamic Pricing Analysis

A Python package for analyzing dynamic pricing strategies in oligopolistic markets, comparing traditional optimization and heuristic approaches with reinforcement learning methods.

## Overview

This package provides tools for simulating and analyzing different pricing strategies in markets with multiple competing firms. It implements:

- Traditional pricing approaches based on first-order conditions (FOC)
- Heuristic pricing strategies
- Reinforcement learning-based pricing strategies
- Market equilibrium analysis
- Tools for analyzing competition and market dynamics


## Quick Start

```python
from dynamic_pricing_sim import Market, FOCFirm, HeuristicFirm, RLFirm
import numpy as np

# Set up market parameters
demand_params = {
    'alpha': 100,  # demand intercept
    'beta': 2,    # own-price effect
    'gamma': 1,    # cross-price effect
    'xi': 0.5,  # own-delivery time effect
    'rho': 0.25  # cross-delivery time effect
}

# Create market with 3 firms
market = Market(N=3, demand_params=demand_params)

# Create firms with different strategies
firms = [
    FOCFirm(cost=20, market=market),
    HeuristicFirm(cost=20, markup=0.1),
    RLFirm(state_size=7, action_size=100, cost=20)
]

# Run simulation
from dynamic_pricing_analysis import run_simulation, plot_results

delivery_times = np.array([1, 2, 3])
prices_history, profits_history = run_simulation(
    T=1000,  # time steps
    market=market,
    firms=firms,
    delivery_times=delivery_times
)

# Plot results
plot_results(prices_history, profits_history, ['FOC', 'Heuristic', 'RL'])
```

## Package Structure

```
src/
├── __init__.py
├── market.py       # Market environment
├── demand.py       # Demand functions
├── equilibrium.py  # Equilibrium solver
├── firms/         # Different firm types
│   ├── __init__.py
│   ├── base_firm.py
│   ├── foc_firm.py
│   ├── heuristic_firm.py
│   └── rl_firm.py
└── simulation.py   # Simulation runner, viz
```

## Components

### Market

The `Market` class represents the market environment:

```python
market = Market(N=3, demand_params={
    'alpha': 100, 'beta': 2, 'gamma': 1, 'xi': 0.5, 'rho': 0.25
})
```

### Firms

Three types of firms are available:

1. **FOC-based Firm**: Uses first-order conditions to set prices
```python
foc_firm = FOCFirm(cost=20, market=market)
```

2. **Heuristic Firm**: Uses simple markup strategy
```python
heuristic_firm = HeuristicFirm(cost=20, markup=0.1)
```

3. **RL Firm**: Uses reinforcement learning to adapt pricing
```python
rl_firm = RLFirm(state_size=7, action_size=100, cost=20)
```

### Simulation

Run market simulations with multiple firms:

```python
prices, profits = run_simulation(
    T=1000,
    market=market,
    firms=firms,
    delivery_times=delivery_times
)
```

## Advanced Usage

### Custom Demand Functions

Create custom demand functions by subclassing `Demand`:

```python
from dynamic_pricing_analysis import Demand

class CustomDemand(Demand):
    def calculate(self, prices, delivery_times):
        # Custom demand calculation
        return demands
```

### Custom Firm Types

Create custom firm types by subclassing `BaseFirm`:

```python
from dynamic_pricing_analysis.firms import BaseFirm

class CustomFirm(BaseFirm):
    def set_price(self, state):
        # Custom pricing logic
        return price

    def optimal_price(self, prices, demands, firm_index):
        # Custom optimization logic
        return optimal_price
```