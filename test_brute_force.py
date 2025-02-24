import numpy as np
from src.utils.brute_force import BruteForceOptimizer
from src.utils.bilp_optimizers import BinaryProgramSolver
import time
from functools import partial

def objective(x, weights):
    return np.sum(weights * x)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    N = 10  # vector length
    C = (2, 4)  # cardinality constraints
    num_cores = 4  # number of CPU cores to use
    
    # Generate random weights
    weights = np.random.uniform(0, 1, N)
    print("Weights:", weights)
    
    # Test brute force method
    print("\n=== Brute Force Method ===")
    bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=num_cores)
    
    start_time = time.time()
    bf_x, bf_val = bf_optimizer.maximize(partial(objective, weights=weights))
    bf_time = time.time() - start_time
    
    print(f"Optimal solution: {bf_x}")
    print(f"Selected indices: {np.where(bf_x == 1)[0]}")
    print(f"Number of selected items: {np.sum(bf_x)}")
    print(f"Optimal value: {bf_val:.4f}")
    print(f"Computation time: {bf_time:.4f} seconds")
    
    # Test BILP method
    print("\n=== BILP Method ===")
    bilp_solver = BinaryProgramSolver()
    
    # Create constraints matrix for cardinality constraints
    A = np.vstack([
        np.ones(N),   # upper bound: sum(x) <= C[1]
        -np.ones(N)   # lower bound: -sum(x) <= -C[0] (equivalent to sum(x) >= C[0])
    ])
    b = np.array([C[1], -C[0]])
    
    start_time = time.time()
    bilp_val, bilp_x, status = bilp_solver.maximize(weights, A, b)
    bilp_time = time.time() - start_time
    
    print(f"Status: {status}")
    print(f"Optimal solution: {bilp_x}")
    print(f"Selected indices: {np.where(bilp_x == 1)[0]}")
    print(f"Number of selected items: {np.sum(bilp_x)}")
    print(f"Optimal value: {bilp_val:.4f}")
    print(f"Computation time: {bilp_time:.4f} seconds")
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"Solutions match: {np.allclose(bf_x, bilp_x)}")
    print(f"Values match: {np.allclose(bf_val, bilp_val)}")
    print(f"Speed ratio (BILP/BF): {bilp_time/bf_time:.2f}x") 