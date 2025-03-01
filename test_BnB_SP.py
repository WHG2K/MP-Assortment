import numpy as np
from scipy.special import softmax
from src.utils.brute_force import BruteForceOptimizer
from src.algorithms.models import MPAssortOriginal, MPAssortSurrogate
from src.utils.distributions import GumBel
from src.utils.bilp_optimizers import BinaryProgramSolver
import time
from src.algorithms.BB import branch_and_bound
from src.utils.brute_force import BruteForceOptimizer



'''
Branch and Bound for SP
'''

class SP_obj:
    def __init__(self, model):
        self.model = model

    def __call__(self, w):
        return self.model.SP(w, solver='gurobi')[1]

class SP_ub:
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

        bilp_solver = BinaryProgramSolver(solver='gurobi')
        upper_bound, _, status = bilp_solver.maximize(c, A, b)
        if status != 'Optimal':
            raise ValueError(f"Failed to solve SP upper bound: {status}")
        return upper_bound
    
class SP_lb:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):
        box_middle = (box_low + box_high) / 2
        w_list = [box_low, box_high, box_middle]
        w_best = None
        val_best = -np.inf
        for w in w_list:
            val = self.model.SP(w, solver='gurobi')[1]
            if val > val_best:
                val_best = val
                w_best = w
        return val_best, w_best

    


if __name__ == "__main__":
    
    np.random.seed(2025)

    for _ in range(3):

        #### Problem parameters ####
        N = 15  # Number of products
        C = (8, 8)  # Cardinality constraints

        # Generate random problem instance
        u = np.random.normal(0, 1, N)
        eu = np.exp(u)
        eu_max = np.max(eu)
        r = eu_max - eu

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

        ########################################
        ##### Branch and Bound to solve SP #####
        ########################################
        w_range = np.array(sp._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)
        print("box_low", np.round(box_low, 2))
        print("box_hig", np.round(box_high, 2))

        sp_obj = SP_obj(sp)
        sp_ub = SP_ub(sp)
        sp_lb = SP_lb(sp)

        t_start = time.time()
        # Run branch and bound algorithm
        w_sp, best_objective = branch_and_bound(
                                    sp_obj, sp_lb, sp_ub, 
                                    box_low, box_high, 
                                    tolerance=0.05, 
                                    min_box_size=0.03,
                                    num_workers=8
                                )
        t_end = time.time()
        print(f"BnB computation time: {t_end - t_start:.2f} seconds")

        print(f"Optimal solution: {w_sp}")
        print(f"Optimal objective value: {best_objective}")
        



        ###################################
        ##### Brute Force to solve OP #####
        ###################################
        num_cores = 8
        bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=num_cores)

        start_time = time.time()
        x_op, val_op = bf_optimizer.maximize(op)
        time_op = time.time() - start_time

        print(f"Optimal solution: {x_op}")
        print(f"Selected indices: {np.where(x_op == 1)[0]}")
        print(f"Optimal value: {val_op:.4f}")
        print(f"Computation time: {time_op:.4f} seconds")


        start_time = time.time()
        x_sp, _ = sp.SP(w_sp)
        time_sp = time.time() - start_time

        print(f"Optimal solution: {x_sp}")
        print(f"Optimal value: {op(x_sp):.4f}")
        print(f"Computation time: {time_sp:.4f} seconds")

        # Compare solutions under original objective
        print("\n=== Solution Comparison under Original Pi ===")
        op_val_for_x_op = op(x_op)
        op_val_for_x_sp = op(x_sp)

        print(f"Original Pi value for x_op: {op_val_for_x_op:.4f}")
        print(f"Original Pi value for x_sp: {op_val_for_x_sp:.4f}")
        print(f"Relative gap: {(op_val_for_x_op - op_val_for_x_sp)/op_val_for_x_op:.4%}")
