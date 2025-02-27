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




'''
rsp objective functions
'''

class rsp_obj:
    def __init__(self, model):
        self.model = model

    def __call__(self, w):
        return self.model.RSP(w)
    
class rsp_ub:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):

        # Compute objective coefficients c
        c = self.model._probs_buying_surrogate(box_low) * self.model.r
        
        # Construct constraint matrix A
        A = np.vstack([
            self.model._probs_U_exceed_w(box_high),  # First |B| rows are P(u[j] + X > w[i])
            np.ones(self.N),   # Cardinality upper bound
            -np.ones(self.N)   # Cardinality lower bound
        ])
        
        # Compute RHS vector b
        b = np.concatenate([
            self.model._Bs,
            [self.model.C[1], -self.model.C[0]]
        ])

        lp_solver = LinearProgramSolver(c, A, b)
        upper_bound, _, status = lp_solver.maximize(c, A, b)
        if status != 'Optimal':
            raise ValueError(f"Failed to solve RSP upper bound: {status}")
        return upper_bound
    
class rsp_lb:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):
        box_middle = (box_low + box_high) / 2
        rsp_box_low = self.model.RSP(box_low)
        rsp_box_high = self.model.RSP(box_high)
        rsp_box_middle = self.model.RSP(box_middle)

        return max(rsp_box_low, rsp_box_middle, rsp_box_high)
    


if __name__ == "__main__":

    # a = np.array([1, 2, 3])
    # print(a ** 2)
    
    box_low = np.array([-1, -1, -1], dtype=float)  # Lower bounds of objective function
    box_high = np.array([2, 2, 2], dtype=float)  # Upper bounds of objective function

    # Run branch and bound algorithm
    best_solution, best_objective = branch_and_bound(test_obj, test_lb, test_ub, box_low, box_high)

    print(f"Optimal solution: {best_solution}")
    print(f"Optimal objective value: {best_objective}")
    