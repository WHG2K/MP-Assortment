import numpy as np
# from scipy.special import softmax
from src.utils.distributions import GumBel
import pickle
from tqdm import tqdm
import time
from src.algorithms.models import MPAssortSurrogate, MPAssortOriginal
from src.algorithms.sBB import spatial_branch_and_bound_maximize
from src.utils.brute_force import BruteForceOptimizer
from src.utils.greedy import GreedyOptimizer
from src.algorithms.models import MNL
from src.algorithms.sBB_functions_utils import RSP_obj, RSP_ub, RSP_lb, SP_obj, SP_ub, SP_lb, OP_obj
from dotenv import load_dotenv
import os
load_dotenv(override=True)



if __name__ == "__main__":

    np.random.seed(0)


    for sample_id in range(100):

        N = 30
        u = np.random.normal(0, 1, N)
        w = np.exp(u)
        w_max = np.max(w)
        r = w_max - w
        
        # Generate basket size distribution
        basket_sizes = [1, 2, 3, 4, 5]
        probs = np.random.uniform(0, 1, len(basket_sizes))
        probs = probs / probs.sum()
        B = dict(zip(basket_sizes, probs))

        C = (15, 20)

        distr = GumBel()
        
        
        sp = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
        w_range = np.array(sp._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)

        w = np.random.uniform(box_low, box_high)
        x_rsp, val = sp.RSP(w, solver='gurobi')

        print(f"sample {sample_id}: {np.round(x_rsp, 4)}")

        c = np.sum([(0.0001 < x_rsp[i] < 0.9999) for i in range(N)])
        if (c > 2):
            print("EXCEPTION FOUND.")
            break

        # if sample_id == 3:
        #     print(np.round(x_rsp, 4))

        #     c = np.array(sp._probs_buying_surrogate(w)).reshape(-1)
        #     d = np.array(sp._probs_U_exceed_w(w)).reshape(-1)

        #     print(c)
        #     print(d)

        #     print(r[2]*c[2]/d[2])
        #     print(r[14]*c[14]/d[14])

        
