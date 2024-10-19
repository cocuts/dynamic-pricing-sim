import numpy as np

class Demand:
    def __init__(self, alpha, beta, gamma, xi, rho):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.rho = rho

    def calculate(self, prices, delivery_times):
        N = len(prices)
        D = np.zeros(N)
        for i in range(N):
            D[i] = self.alpha - self.beta * prices[i] + self.gamma * sum(prices) - self.xi * delivery_times[i] + self.rho * sum(delivery_times)
        return np.maximum(D, 0)  # Ensure non-negative demand