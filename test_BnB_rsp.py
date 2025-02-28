import numpy as np
from scipy.special import softmax
from src.utils.brute_force import BruteForceOptimizer
from src.algorithms.models import MPAssortOriginal, MPAssortSurrogate
from src.utils.distributions import GumBel
from src.utils.lp_optimizers import LinearProgramSolver
import time
from src.algorithms.BB import branch_and_bound
from src.utils.brute_force import BruteForceOptimizer



'''
Branch and Bound for RSP
'''

class RSP_obj:
    def __init__(self, model):
        self.model = model

    def __call__(self, w):
        return self.model.RSP(w, solver='gurobi')[1]

class RSP_ub:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):

        N = len(self.model.u)

        # Compute objective coefficients c
        c = self.model._probs_buying_surrogate(box_low) * self.model.r
        
        # Construct constraint matrix A
        A = np.vstack([
            self.model._probs_U_exceed_w(box_high),  # First |B| rows are P(u[j] + X > w[i])
            np.ones(N),   # Cardinality upper bound
            -np.ones(N)   # Cardinality lower bound
        ])
        
        # Compute RHS vector b
        b = np.concatenate([
            self.model._Bs,
            [self.model.C[1], -self.model.C[0]]
        ])

        lp_solver = LinearProgramSolver(solver='gurobi')
        upper_bound, _, status = lp_solver.maximize(c, A, b)
        if status != 'Optimal':
            raise ValueError(f"Failed to solve RSP upper bound: {status}")
        return upper_bound
    
class RSP_lb:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):
        box_middle = (box_low + box_high) / 2
        # rsp_box_low = self.model.RSP(box_low)
        # rsp_box_high = self.model.RSP(box_high)
        rsp_box_middle = self.model.RSP(box_middle, solver='gurobi')[1]
        return rsp_box_middle, box_middle

    


if __name__ == "__main__":
    
    #### Problem parameters ####
    np.random.seed(2025)
    N = 60  # Number of products
    C = (12, 12)  # Cardinality constraints

    # Generate random problem instance
    # np.random.seed(42)
    u = np.random.normal(0, 1, N)
    eu = np.exp(u)
    eu_max = np.max(eu)
    r = eu_max - eu
    # r = np.random.uniform(1, 10, N)


    # Generate basket size distribution
    basket_sizes = [1, 2, 3]
    probs = np.random.normal(0, 1, len(basket_sizes))
    probs = softmax(probs)
    B = dict(zip(basket_sizes, probs))

    # Create distribution
    distr = GumBel()

    # Create objective functions
    n_samples = 10000
    op = MPAssortOriginal(u, r, B, distr, C, samples=distr.random_sample((n_samples, len(u)+1)))
    sp = MPAssortSurrogate(u, r, B, distr, C)

    #########################################
    ##### Branch and Bound to solve RSP #####
    #########################################
    w_range = np.array(sp._get_box_constraints())
    box_low = np.array(w_range[:, 0]).reshape(-1)
    box_high = np.array(w_range[:, 1]).reshape(-1)
    print("box_low", np.round(box_low, 2))
    print("box_hig", np.round(box_high, 2))

    rsp_obj = RSP_obj(sp)
    rsp_ub = RSP_ub(sp)
    rsp_lb = RSP_lb(sp)

    t_start = time.time()
    # Run branch and bound algorithm
    w_rsp, best_objective = branch_and_bound(
                                rsp_obj, rsp_lb, rsp_ub, 
                                box_low, box_high, 
                                tolerance=0.05, 
                                min_box_size=0.01,
                                num_workers=8
                            )
    t_end = time.time()
    print(f"BnB computation time: {t_end - t_start:.2f} seconds")

    print(f"Optimal solution: {w_rsp}")
    print(f"Optimal objective value: {best_objective}")
    



    ###################################
    ##### Brute Force to solve OP #####
    ###################################
    num_cores = 4
    bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=num_cores)

    start_time = time.time()
    x_op, val_op = bf_optimizer.maximize(op)
    time_op = time.time() - start_time

    print(f"Optimal solution: {x_op}")
    print(f"Selected indices: {np.where(x_op == 1)[0]}")
    print(f"Optimal value: {val_op:.4f}")
    print(f"Computation time: {time_op:.4f} seconds")


    start_time = time.time()
    x_rsp, _ = sp.SP(w_rsp)
    time_sp = time.time() - start_time

    print(f"Optimal solution: {x_rsp}")
    print(f"Selected indices: {np.where(x_rsp == 1)[0]}")
    print(f"Optimal value: {op(x_rsp):.4f}")
    print(f"Computation time: {time_sp:.4f} seconds")

    # Compare solutions under original objective
    print("\n=== Solution Comparison under Original Pi ===")
    op_val_for_x_op = op(x_op)
    op_val_for_x_sp = op(x_rsp)

    print(f"Original Pi value for x_op: {op_val_for_x_op:.4f}")
    print(f"Original Pi value for x_sp: {op_val_for_x_sp:.4f}")
    print(f"Relative gap: {(op_val_for_x_op - op_val_for_x_sp)/op_val_for_x_op:.4%}")
