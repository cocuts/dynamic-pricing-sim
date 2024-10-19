import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
from .base_firm import BaseFirm

@dataclass
class RLHyperParams:
    """Hyperparameters for RL agent configuration"""
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
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, params.hidden_size))
        if params.batch_norm:
            layers.append(nn.BatchNorm1d(params.hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(params.n_hidden_layers - 1):
            layers.append(nn.Linear(params.hidden_size, params.hidden_size))
            if params.batch_norm:
                layers.append(nn.BatchNorm1d(params.hidden_size))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(params.hidden_size, action_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class RLFirm(BaseFirm):
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 cost: float,
                 market: Optional = None,
                 hyperparams: Optional[RLHyperParams] = None):
        """
        Initialize RL-based firm
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            cost: Production cost
            market: Optional market object for accessing demand parameters
            hyperparams: Optional hyperparameters, uses defaults if None
        """
        super().__init__(cost)
        self.state_size = state_size
        self.action_size = action_size
        self.market = market
        
        # Use provided hyperparams or defaults
        self.params = hyperparams if hyperparams is not None else RLHyperParams()
        
        # Initialize DQN and training components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size, self.params).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        
        # Initialize exploration parameters
        self.epsilon = self.params.epsilon
        self.epsilon_min = self.params.epsilon_min
        self.epsilon_decay = self.params.epsilon_decay
        self.gamma = self.params.gamma

    def set_price(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            act_values = self.model(state_tensor)
            return int(torch.argmax(act_values[0]).cpu().numpy())

    def optimal_price(self, prices, demands, firm_index):
        state = np.array([0] + list(prices) + [0] * (self.state_size - len(prices) - 1)).reshape(1, -1)
        action = self.set_price(state)
        
        # If market is provided, use its parameters for price bounds
        if self.market:
            max_price = self.market.demand.alpha / self.market.demand.beta
        else:
            max_price = np.max(prices) if len(prices) > 0 else 100
            
        return action / self.action_size * (max_price - self.cost) + self.cost

    def train(self, state, action, reward, next_state, done):
        self.model.train()
        
        # Bound action to valid range
        action = min(max(action, 0), self.action_size - 1)
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Calculate target Q-value
        with torch.no_grad():
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)[0])
            else:
                target = reward_tensor

        # Update Q-values
        current_q_values = self.model(state_tensor)
        target_q_values = current_q_values.clone()
        target_q_values[0][action] = target

        # Compute loss and update weights
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay