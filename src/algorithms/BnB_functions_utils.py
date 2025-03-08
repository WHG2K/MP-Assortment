from src.utils.lp_optimizers import LinearProgramSolver
from src.utils.bilp_optimizers import BinaryProgramSolver
import numpy as np


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

    # def __call__(self, box_low, box_high):
    #     box_middle = (box_low + box_high) / 2
    #     # rsp_box_low = self.model.RSP(box_low)
    #     # rsp_box_high = self.model.RSP(box_high)
    #     rsp_box_middle = self.model.RSP(box_middle, solver='gurobi')[1]
    #     return rsp_box_middle, box_middle

    def __call__(self, box_low, box_high):
        box_middle = (box_low + box_high) / 2
        x_list = [box_low, box_high, box_middle]
        x_best = None
        val_best = -np.inf
        for x in x_list:
            val = self.model.RSP(x, solver='gurobi')[1]
            if val > val_best:
                val_best = val
                x_best = x
        return val_best, x_best
    

class OP_obj:
    def __init__(self, model, random_comps):
        self.model = model
        self.random_comps = random_comps

    def __call__(self, x):
        return self.model._pi_monte_carlo(x, random_comps=self.random_comps)
