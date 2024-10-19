from abc import ABC, abstractmethod

class BaseFirm(ABC):
    def __init__(self, cost):
        self.cost = cost

    @abstractmethod
    def set_price(self, state):
        pass

    @abstractmethod
    def optimal_price(self, prices, demands, firm_index):
        pass