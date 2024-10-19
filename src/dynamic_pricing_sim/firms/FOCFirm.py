import numpy as np
from .base_firm import BaseFirm

class FOCFirm(BaseFirm):
    def __init__(self, cost, market):
        super().__init__(cost)
        self.market = market

    def set_price(self, state):
        other_prices = state[1:self.market.N+1]
        avg_other_price = np.mean(other_prices)
        price = (self.market.demand.alpha + self.market.demand.beta * self.cost + 
                self.market.demand.gamma * (self.market.N - 1) * avg_other_price) / (2 * self.market.demand.beta)
        return max(price, self.cost)

    def optimal_price(self, prices, demands, firm_index):
        other_prices = np.concatenate([prices[:firm_index], prices[firm_index+1:]])
        avg_other_price = np.mean(other_prices)
        return (self.market.demand.alpha + self.market.demand.beta * self.cost + 
                self.market.demand.gamma * (self.market.N - 1) * avg_other_price) / (2 * self.market.demand.beta)