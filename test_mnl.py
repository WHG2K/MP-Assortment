import numpy as np
from src.algorithms.models import MNL
from src.utils.brute_force import BruteForceOptimizer
import time
from dotenv import load_dotenv
import os
load_dotenv(override=True)


if __name__ == "__main__":
    # test_mnl_vs_brute_force() 

    np.random.seed(42)
    
    # Problem parameters
    N = 10  # number of products
    C = (5, 7)  # cardinality constraints
    
    # Generate random instance
    u = np.random.normal(0, 1, N)  # utilities
    r = np.random.uniform(1, 10, N)  # revenues
    
    # Create MNL instance and solve
    mnl = MNL(u, r)
    start_time = time.time()
    x_mnl, val_mnl = mnl.solve(C)
    time_mnl = time.time() - start_time
    
    # Solve with brute force
    bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=8)
    start_time = time.time()
    x_bf, val_bf = bf_optimizer.maximize(mnl)
    time_bf = time.time() - start_time
    
    # Print results
    print("\n=== Problem Parameters ===")
    print(f"N = {N}")
    print(f"C = {C}")
    print(f"\nu = {u.round(4)}")
    print(f"r = {r.round(4)}")
    
    print("\n=== MNL Solution ===")
    print(f"x = {x_mnl}")
    print(f"obj = {val_mnl:.6f}")
    print(f"time = {time_mnl:.4f} seconds")
    
    print("\n=== Brute Force Solution ===")
    print(f"x = {x_bf}")
    print(f"obj = {val_bf:.6f}")
    print(f"time = {time_bf:.4f} seconds")

    # x = np.random.randint(0, 2, N)
    # print(x)
    # print(mnl(x))