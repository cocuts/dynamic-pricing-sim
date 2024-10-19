import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .firms import RLFirm
from .equilibrium import Equilibrium
import pandas as pd

def run_simulation(T, market, firms, delivery_times):
    N = market.N
    prices_history = np.zeros((T, N))
    profits_history = np.zeros((T, N))
    equilibrium = Equilibrium(market, firms)
    
    for t in range(T):
        equilibrium_prices, equilibrium_demands = equilibrium.find_equilibrium(delivery_times)
        profits = equilibrium.calculate_profits(equilibrium_prices, equilibrium_demands)
        
        state = np.array([t] + list(equilibrium_prices) + list(delivery_times)).reshape(1, -1)
        for i, firm in enumerate(firms):
            if isinstance(firm, RLFirm.RLFirm):
                next_state = np.array([t+1] + list(equilibrium_prices) + list(delivery_times)).reshape(1, -1)
                
                # Ensure action index is within bounds
                action_index = int(equilibrium_prices[i] * firm.action_size / 
                                 (market.demand.alpha / market.demand.beta))
                action_index = min(action_index, firm.action_size - 1)  # Ensure we don't exceed action_size
                action_index = max(0, action_index)  # Ensure we don't go below 0
                
                firm.train(state, action_index, profits[i], next_state, t == T-1)
        
        prices_history[t] = equilibrium_prices
        profits_history[t] = profits
    
    return prices_history, profits_history

def plot_results(prices_history, profits_history, firm_types):
    T, N = prices_history.shape
    
    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Convert data to long format for seaborn
    time_points = np.arange(T)
    
    # Prepare data for prices plot
    prices_data = []
    for i in range(N):
        for t in range(T):
            prices_data.append({
                'Time': t,
                'Price': prices_history[t, i],
                'Firm Type': firm_types[i]
            })
    prices_df = pd.DataFrame(prices_data)
    
    # Prepare data for profits plot
    profits_data = []
    for i in range(N):
        for t in range(T):
            profits_data.append({
                'Time': t,
                'Profit': profits_history[t, i],
                'Firm Type': firm_types[i]
            })
    profits_df = pd.DataFrame(profits_data)
    
    # Plot prices
    sns.lineplot(data=prices_df, x='Time', y='Price', hue='Firm Type', ax=ax1)
    ax1.set_title('Price Dynamics', pad=20, fontsize=14)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    
    # Plot profits
    sns.lineplot(data=profits_df, x='Time', y='Profit', hue='Firm Type', ax=ax2)
    ax2.set_title('Profit Dynamics', pad=20, fontsize=14)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Profit', fontsize=12)
    
    # Adjust layout and styling
    plt.tight_layout()
    
    return fig