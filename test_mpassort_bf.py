import numpy as np
from scipy.special import softmax
from src.utils.brute_force import BruteForceOptimizer
from src.algorithms.models import MPAssortOriginal, MPAssortSurrogate
from src.utils.distributions import GumBel
from src.utils.lp_optimizers import LinearProgramSolver
import time



'''
rsp objective functions
'''

class RSP_obj:
    def __init__(self, model):
        self.model = model

    def __call__(self, w):
        return self.model.RSP(w)
    
class RSP_ub:
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
    
class RSP_lb:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):
        box_middle = (box_low + box_high) / 2
        rsp_box_low = self.model.RSP(box_low)
        rsp_box_high = self.model.RSP(box_high)
        rsp_box_middle = self.model.RSP(box_middle)

        return max(rsp_box_low, rsp_box_middle, rsp_box_high)






if __name__ == "__main__":
    # Problem parameters
    N = 10  # Number of products
    C = (2, 4)  # Cardinality constraints
    num_cores = 4
    
    # Generate random problem instance
    np.random.seed(42)
    u = np.random.normal(0, 1, N)
    r = np.random.uniform(1, 10, N)
    
    # Generate basket size distribution
    basket_sizes = [1, 2, 3]
    probs = np.random.normal(0, 1, len(basket_sizes))
    probs = softmax(probs)
    B = dict(zip(basket_sizes, probs))
    
    # Create distribution
    distr = GumBel()
    
    # Create objective functions
    n_samples = 10000
    obj_op = MPAssortOriginal(u, r, B, distr, C, samples=distr.random_sample((n_samples, len(u)+1)))
    obj_sp = MPAssortSurrogate(u, r, B, distr, C)
    
    # Solve using brute force
    print("=== Brute Force with Original Pi ===")
    bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=num_cores)
    
    start_time = time.time()
    x_op, val_op = bf_optimizer.maximize(obj_op)
    time_op = time.time() - start_time
    
    print(f"Optimal solution: {x_op}")
    print(f"Selected indices: {np.where(x_op == 1)[0]}")
    print(f"Number of selected items: {np.sum(x_op)}")
    print(f"Optimal value: {val_op:.4f}")
    print(f"Computation time: {time_op:.4f} seconds")
    
    print("\n=== Brute Force with Surrogate Pi ===")
    start_time = time.time()
    x_sp, val_sp = bf_optimizer.maximize(obj_sp)
    time_sp = time.time() - start_time
    
    print(f"Optimal solution: {x_sp}")
    print(f"Selected indices: {np.where(x_sp == 1)[0]}")
    print(f"Number of selected items: {np.sum(x_sp)}")
    print(f"Optimal value: {val_sp:.4f}")
    print(f"Computation time: {time_sp:.4f} seconds")
    
    # Compare solutions under original objective
    print("\n=== Solution Comparison under Original Pi ===")
    op_val_for_x_op = obj_op(x_op)
    op_val_for_x_sp = obj_op(x_sp)
    
    print(f"Original Pi value for x_op: {op_val_for_x_op:.4f}")
    print(f"Original Pi value for x_sp: {op_val_for_x_sp:.4f}")
    print(f"Relative gap: {(op_val_for_x_op - op_val_for_x_sp)/op_val_for_x_op:.4%}") 