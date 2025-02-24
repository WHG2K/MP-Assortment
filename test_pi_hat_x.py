import numpy as np
from scipy.special import softmax
from src.utils.distributions import NorMal
from src.algorithms.solvers import MPAssortSurrogate

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Problem parameters
    N = 15  # Number of products
    u = np.random.normal(0, 1, N)
    r = np.random.uniform(1, 10, N)
    
    # Generate basket size distribution
    basket_sizes = [1, 2, 3]
    probs = np.random.normal(0, 1, len(basket_sizes))
    probs = softmax(probs)
    B = dict(zip(basket_sizes, probs))
    
    # Other parameters
    distr = NorMal(std=1.0)
    C = (3, 7)
    
    # Create solver instance
    algo = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
    
    # Create test assortments
    x1 = np.zeros(N)
    x1[0:5] = 1  # Select first 5 products
    
    x2 = np.zeros(N)
    x2[5:10] = 1  # Select products 5-9
    
    # Test _pi_hat_x for both assortments
    print("Testing π_hat(x) computation...")
    print("\nAssortment 1:")
    print(f"x1: {x1}")
    pi_hat_x1 = algo._pi_hat_x(x1)
    print(f"π_hat(x1): {pi_hat_x1:.4f}")
    
    print("\nAssortment 2:")
    print(f"x2: {x2}")
    pi_hat_x2 = algo._pi_hat_x(x2)
    print(f"π_hat(x2): {pi_hat_x2:.4f}")
    
    # Print the revenues for comparison
    print("\nRevenues of selected products:")
    print(f"x1 selected products revenues: {r[0:5]}")
    print(f"x2 selected products revenues: {r[5:10]}") 