import numpy as np
from scipy.optimize import fsolve

class Equilibrium:
    def __init__(self, market, firms):
        self.market = market
        self.firms = firms

    def find_equilibrium(self, delivery_times):
        def equations(prices):
            demands = self.market.get_demand(prices, delivery_times)
            return [firm.optimal_price(prices, demands, i) - prices[i] for i, firm in enumerate(self.firms)]

        initial_guess = [firm.cost for firm in self.firms]
        equilibrium_prices = fsolve(equations, initial_guess)
        equilibrium_demands = self.market.get_demand(equilibrium_prices, delivery_times)
        
        return equilibrium_prices, equilibrium_demands

    def calculate_profits(self, prices, demands):
        return [(prices[i] - self.firms[i].cost) * demands[i] for i in range(len(self.firms))]