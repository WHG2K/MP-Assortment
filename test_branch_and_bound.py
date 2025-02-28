import numpy as np
from src.utils.lp_optimizers import LinearProgramSolver
from collections import deque
from src.algorithms.BB import branch_and_bound



'''
A test example for branch and bound
'''

# Define objective function
def test_obj(x):
    x = np.array(x, dtype=float).reshape(-1)
    return -np.sum(x ** 2) # Maximization problem, take negative

# Define lower bound function (minimum possible value)
def test_lb(lower_bounds, upper_bounds):
    lower_bounds = np.array(lower_bounds, dtype=float).reshape(-1)
    upper_bounds = np.array(upper_bounds, dtype=float).reshape(-1)
    # Simple implementation: use objective value at lower bounds
    return -np.sum(lower_bounds ** 2), lower_bounds

# Define upper bound function (maximum possible value)
def test_ub(lower_bounds, upper_bounds):
    lower_bounds = np.array(lower_bounds, dtype=float).reshape(-1)
    upper_bounds = np.array(upper_bounds, dtype=float).reshape(-1)
    # Simple implementation: use objective value at upper bounds
    a = np.zeros(len(lower_bounds), dtype=float)
    for i in range(len(lower_bounds)):
        if (lower_bounds[i] * upper_bounds[i]) > 0:
            a[i] = min(abs(lower_bounds[i]), abs(upper_bounds[i]))
        else:
            a[i] = 0
    return -np.sum(a ** 2)

    


if __name__ == "__main__":
    
    box_low = np.array([-1, -1, -1], dtype=float)  # Lower bounds of objective function
    box_high = np.array([2, 2, 2], dtype=float)  # Upper bounds of objective function

    # Run branch and bound algorithm
    best_solution, best_objective = branch_and_bound(test_obj, test_lb, test_ub, box_low, box_high, tolerance=1e-2, min_box_size=5e-2)

    print(f"Optimal solution: {best_solution}")
    print(f"Optimal objective value: {best_objective}")
    