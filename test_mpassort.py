import numpy as np
from scipy.special import softmax
from src.utils.distributions import GumBel
from algorithms.models import MPAssortSurrogate
import time

if __name__ == "__main__":
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
    distr = GumBel()
    C = (3, 7)
    
    # Create solver instance
    algo = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
    
    # Generate w vector from normal distribution
    w = np.random.normal(0, 1, len(B))
    
    # Calculate SP value with timing
    start_time = time.time()
    _, sp_w = algo.SP(w)
    sp_time = time.time() - start_time
    print("=== SP Value ===")
    print(f"SP value: {sp_w:.4f}")
    print(f"Computation time: {sp_time:.4f} seconds")

    # Calculate RSP value with timing
    start_time = time.time()
    _, rsp_w = algo.RSP(w)
    rsp_time = time.time() - start_time
    print("\n=== RSP Value ===")
    print(f"RSP value: {rsp_w:.4f}")
    print(f"Computation time: {rsp_time:.4f} seconds")
    
    # Compare times
    print("\n=== Time Comparison ===")
    print(f"RSP/SP time ratio: {rsp_time/sp_time:.2f}x")
    