import numpy as np
from src.utils.distributions import NorMal
from src.algorithms.solvers import MPAssortOriginal

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a small test case
    N = 5  # Small number of products for easy verification
    u = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    r = np.random.uniform(1, 10, N)  # Random revenues from [1,10]
    B = {1: 0.6, 2: 0.4}  # Single basket size for simplicity
    distr = NorMal(std=1.0)
    C = (1, 2)
    
    # Create solver instance
    algo = MPAssortOriginal(u=u, r=r, B=B, distr=distr, C=C)
    
    # Create test assortment
    x = np.array([1, 0, 1, 0, 1])  # Select first two products
    
    # Generate random samples
    n_samples = 5  # Small number for checking intermediate values
    random_comps = algo.generate_samples(n_samples)
    
    # Print revenues for reference
    print("Revenues:", r)
    
    # Test _pi_monte_carlo
    print("\nTesting _pi_monte_carlo...")
    pi_x_monte_carlo = algo._pi_monte_carlo(x, random_comps) 
    print(f"pi_x_monte_carlo: {pi_x_monte_carlo:.4f}")