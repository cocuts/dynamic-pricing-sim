import numpy as np
from .base_firm import BaseFirm

class HeuristicFirm(BaseFirm):
    def __init__(self, cost, markup):
        super().__init__(cost)
        self.markup = markup

    def set_price(self, state):
        competitor_prices = state[1:-1]  # Assuming state includes time and delivery times
        return max(np.mean(competitor_prices) * (1 + self.markup), self.cost)

    def optimal_price(self, prices, demands, firm_index):
        competitor_prices = prices[:firm_index] + prices[firm_index+1:]
        return max(np.mean(competitor_prices) * (1 + self.markup), self.cost)
