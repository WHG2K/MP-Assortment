import numpy as np
from src.utils.nonconvex_optimizer import NonconvexOptimizer
import time

def objective(x):
    return np.sin(x[0]) * np.cos(x[1]) + x[0]/10

if __name__ == "__main__":
    # Problem parameters
    n_dims = 5
    bounds = np.array([[-2, 2]] * n_dims)
    
    # Solver parameters
    solver_params = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 2000}
    }
    
    # Create optimizer
    optimizer = NonconvexOptimizer(
        n_dims=n_dims,
        bounds=bounds,
        solver='BFGS',
        solver_params=solver_params
    )
    
    # Solve with timing
    start_time = time.time()
    x_opt, val_opt = optimizer.maximize(objective)
    solve_time = time.time() - start_time
    
    # Print results
    print("=== Optimization Results ===")
    print(f"Optimal solution: {x_opt}")
    print(f"Optimal value: {val_opt:.6f}")
    print(f"Computation time: {solve_time:.4f} seconds") 