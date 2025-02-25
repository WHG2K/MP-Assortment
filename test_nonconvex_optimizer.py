import numpy as np
from src.utils.nonconvex_optimizer import NonconvexOptimizer


# Test problem: Maximize a nonconvex function
def objective(x):
    return np.sin(x[0]) * np.cos(x[1]) + x[0]/10



if __name__ == "__main__":
    
    # Problem parameters
    n_dims = 2
    bounds = np.array([[-5, 5], [-5, 5]])
    
    # Test all solvers
    solvers = ["BFGS"]
    
    for solver_name in solvers:
        print(f"\n=== Testing {solver_name} ===")
        
        # Create optimizer with specific solver
        optimizer = NonconvexOptimizer(
            n_dims=n_dims,
            bounds=bounds,
            solver=solver_name
        )
        
        # Solve and get detailed results
        results = optimizer.maximize_with_details(objective)
        
        print("Optimization results:")
        for key, value in results.items():
            print(f"{key}: {value}") 