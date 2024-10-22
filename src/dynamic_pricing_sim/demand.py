import numpy as np

class Demand:
    def __init__(self, alpha, beta, gamma, xi, rho):
        """
        Initialize demand system parameters
        alpha: base demand
        beta: own-price sensitivity
        gamma: cross-quantity sensitivity
        xi: own-delivery-time sensitivity
        rho: cross-delivery-time sensitivity
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.rho = rho

    def calculate(self, prices, delivery_times):
        """
        Calculate demand for all firms simultaneously
        
        Parameters:
        prices: vector of firm prices
        delivery_times: vector of firm delivery times
        
        Returns:
        Vector of demands for each firm
        """
        N = len(prices)
        # Form system of equations: AQ = b
        # Where A is the quantity interaction matrix
        # and b is the vector of price and delivery time effects
        
        # Construct A matrix: I - γ*(1-I) where I is identity
        A = np.eye(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    A[i,j] = -self.gamma
        
        # Construct b vector
        b = np.zeros(N)
        for i in range(N):
            other_d = sum(delivery_times[j] for j in range(N) if j != i)
            b[i] = (self.alpha 
                   - self.beta * prices[i] 
                   - self.xi * delivery_times[i]
                   + self.rho * other_d)
        
        # Solve system AQ = b
        try:
            Q = np.linalg.solve(A, b)
            return np.maximum(Q, 0)  # Ensure non-negative demands
        except np.linalg.LinAlgError:
            # If system is singular, return zero demands
            return np.zeros(N)