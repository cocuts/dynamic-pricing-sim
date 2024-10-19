from .demand import Demand

class Market:
    def __init__(self, N, demand_params):
        self.N = N
        self.demand = Demand(**demand_params)

    def get_demand(self, prices, delivery_times):
        return self.demand.calculate(prices, delivery_times)