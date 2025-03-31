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


    for _ in range(1):

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

        C = (15, 15)

        distr = GumBel()
        
        
        sp = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
        w_range = np.array(sp._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)

        #################################################
        #### Solve exact SP via branch-and-bound ########
        #################################################
        if True:
            sp_obj = RSP_obj(sp)
            sp_ub = RSP_ub(sp)
            sp_lb = RSP_lb(sp)

            start_time = time.time()
            w_sp, _ = spatial_branch_and_bound_maximize(
                sp_obj, sp_lb, sp_ub,
                (box_low, box_high),
                tolerance=0.01
            )

            x_sp = sp.SP(w_sp, solver='gurobi')[0]

            end_time = time.time()
            print(f"Exact SP branch-and-bound runtime: {end_time - start_time:.2f} seconds")

        #################################################
        #### Solve exact RSP via branch-and-bound #######
        #################################################

        if True:

            rsp_obj = RSP_obj(sp)
            rsp_ub = RSP_ub(sp)
            rsp_lb = RSP_lb(sp)

            start_time = time.time()
            w_rsp, _ = spatial_branch_and_bound_maximize(
                rsp_obj, rsp_lb, rsp_ub,
                (box_low, box_high),
                tolerance=0.01
            )

            x_rsp = sp.SP(w_rsp, solver='gurobi')[0]

            end_time = time.time()
            print(f"Exact RSP branch-and-bound runtime: {end_time - start_time:.2f} seconds")

        #####################################
        #### Solve OP via brute force #######
        #####################################

        # op = MPAssortOriginal(u, r, B, distr, C)
        # op_obj = OP_obj(op, distr.random_sample((1000, N+1)))

        # bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=24)
        # start_time = time.time()
        # x_op, val_op = bf_optimizer.maximize(op_obj)
        # end_time = time.time()
        # print(f"OP brute-force runtime: {end_time - start_time:.2f} seconds")



        # # calculate gaps
        # gap_sp_exact = 1 - op_obj(x_sp) / op_obj(x_op)
        # gap_rsp_exact = 1 - op_obj(x_rsp) / op_obj(x_op)
        # print(f"Exact SP gap: {gap_sp_exact:.2%}")
        # print(f"Exact RSP gap: {gap_rsp_exact:.2%}")


        
