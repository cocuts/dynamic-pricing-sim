import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
from .base_firm import BaseFirm

@dataclass
class RLHyperParams:
    learning_rate: float = 0.001
    gamma: float = 0.95        # discount rate
    epsilon: float = 1.0       # initial exploration rate
    epsilon_min: float = 0.01  # minimum exploration rate
    epsilon_decay: float = 0.995 # exploration decay rate
    hidden_size: int = 24      # size of hidden layers
    batch_norm: bool = False   # whether to use batch normalization
    n_hidden_layers: int = 2   # number of hidden layers

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, params: RLHyperParams):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, params.hidden_size),
            nn.ReLU(),
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.ReLU(),
            nn.Linear(params.hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)

class RLFirm(BaseFirm):
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 cost: float,
                 market: Optional = None,
                 hyperparams: Optional[RLHyperParams] = None):
        super().__init__(cost)
        self.state_size = state_size
        self.action_size = action_size
        self.market = market
        self.params = hyperparams if hyperparams is not None else RLHyperParams()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size, self.params).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        
        self.epsilon = self.params.epsilon
        self.epsilon_min = self.params.epsilon_min
        self.epsilon_decay = self.params.epsilon_decay
        
        self.min_price = cost * (1 + self.params.min_markup_pct)
        self.last_price = self.min_price  # Initialize at minimum viable price
        self.last_profit = 0
        self.total_profits = []

    def action_to_price(self, action: int, competitor_prices: np.ndarray) -> float:
        """Convert discrete action to price with guaranteed minimum markup"""
        # Ensure action is valid
        action = max(0, min(action, self.action_size - 1))
        
        # Determine price bounds
        if self.market:
            max_price = self.market.demand.alpha / self.market.demand.beta
        else:
            max_price = max(np.max(competitor_prices) * 2, self.cost * 2) if len(competitor_prices) > 0 else self.cost * 2
        
        # Convert action to markup percentage (minimum markup_pct to 100%)
        markup_pct = self.params.min_markup_pct + (action / self.action_size) * (1.0 - self.params.min_markup_pct)
        
        # Calculate price ensuring it's between min_price and max_price
        price = self.cost * (1 + markup_pct)
        return min(max(price, self.min_price), max_price)

    def set_price(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action but constrained to produce valid markup
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            act_values = self.model(state_tensor)
            # Mask out actions that would lead to prices below minimum
            min_action = int(self.params.min_markup_pct * self.action_size)
            act_values[0][:min_action] = float('-inf')
            return int(torch.argmax(act_values[0]).cpu().numpy())

    def optimal_price(self, prices, demands, firm_index):
        # Create state vector matching the expected size
        state = np.zeros(self.state_size)
        state[0] = 0  # time placeholder
        state[1:len(prices)+1] = prices
        if len(prices) + 1 < len(state):
            state[len(prices)+1:] = 0
            
        action = self.set_price(state.reshape(1, -1))
        price = self.action_to_price(action, prices)
        
        self.last_price = price
        return price

    def train(self, state, action, reward, next_state, done):
        self.last_profit = reward
        self.total_profits.append(reward)
        
        self.model.train()
        
        state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).reshape(1, -1).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Using immediate rewards only (gamma = 0)
        target = reward_tensor
        
        current_q_values = self.model(state_tensor)
        target_q_values = current_q_values.clone()
        target_q_values[0][action] = target

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if len(self.total_profits) % 100 == 0:
            recent_profits = self.total_profits[-100:]
            print(f"Recent avg profit: {np.mean(recent_profits):.2f}, "
                  f"Last price: {self.last_price:.2f}, "
                  f"Last profit: {self.last_profit:.2f}")