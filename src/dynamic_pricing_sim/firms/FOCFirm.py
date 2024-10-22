import numpy as np
from .base_firm import BaseFirm

class FOCFirm(BaseFirm):
    def __init__(self, cost, market):
        super().__init__(cost)
        self.market = market

    def optimal_price(self, prices, demands, firm_index):
        other_prices = np.concatenate([prices[:firm_index], prices[firm_index+1:]])
        sum_other_prices = np.sum(other_prices)
        
        # Solving FOC: D + (p-c)(-β) = 0
        # Where D = α - βp + γ sum(p_j)
        # (α - βp + γ sum(p_j)) + (p-c)(-β) = 0
        # α - βp + γ sum(p_j) - βp + βc = 0
        # α + γ sum(p_j )+ βc = 2βp
        price = (self.market.demand.alpha + self.market.demand.gamma * sum_other_prices + 
                self.market.demand.beta * self.cost) / (2 * self.market.demand.beta)
        
        return max(price, self.cost)

    def set_price(self, prices, demands, firm_index):
        other_prices = np.concatenate([prices[:self.market.N//2], prices[self.market.N//2+1:]])
        sum_other_prices = np.sum(other_prices)
        
        price = (self.market.demand.alpha + self.market.demand.gamma * sum_other_prices + 
                self.market.demand.beta * self.cost) / (2 * self.market.demand.beta)
        
        return max(price, self.cost)