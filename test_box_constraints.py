import numpy as np
from scipy.special import softmax
from src.utils.distributions import GumBel
from src.algorithms.solvers import MPAssortSurrogate

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Problem parameters
    N = 15  # Number of products
    C = (3, 7)  # Cardinality constraints
    
    # Generate random problem instance
    u = np.random.normal(0, 1, N)
    r = np.random.uniform(1, 10, N)
    
    # Generate basket size distribution
    basket_sizes = [1, 2, 3]
    probs = np.random.normal(0, 1, len(basket_sizes))
    probs = softmax(probs)
    B = dict(zip(basket_sizes, probs))
    
    # Create distribution
    distr = GumBel()
    
    # Create solver instance
    solver = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
    
    # Print utilities
    print("=== Utilities ===")
    print("Original u:", np.round(u, 2))
    
    # Get rankings (descending order, rank 1 is the highest)
    rankings = np.zeros_like(u, dtype=int)
    rankings[np.argsort(-u)] = np.arange(1, len(u) + 1)
    print("Rankings (1 is highest):", rankings)
    
    # Get box constraints
    bounds = solver._get_box_constraints()
    