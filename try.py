import argparse
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

import pandas as pd


if __name__ == "__main__":
    with open(r'raw_dec_N_60_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001.pkl', 'rb') as f:
        instances = pickle.load(f)

    df = pd.DataFrame(instances)
    # df.to_excel('check_details.xlsx', index=False)
    u = df["u"][0]
    r = df["r"][0]
    # B = df["B"][0]
    B = {2: 0.5, 4: 0.5}
    C = df["C"][0]
    N = len(u)
    distr = GumBel()



    model = MPAssortOriginal(u, r, B, distr, C)

    for _ in range(10):
        x = (np.random.rand(N) < 0.4).astype(int)
        print(x)
        print(model._pi(x))
        print(model._pi_monte_carlo(x, distr.random_sample((10000, N+1))))

    # print(u, r, B, C)

    # #############################################
    # ##### 1. solve exact SP
    # #############################################

    # model = MPAssortSurrogate(u, r, B, distr, C)
    # w_range = np.array(model._get_box_constraints())
    # box_low = np.array(w_range[:, 0]).reshape(-1)
    # box_high = np.array(w_range[:, 1]).reshape(-1)

    # # solve
    # tolerance = 0.0001
    # start_time = time.time()
    # sp_obj = SP_obj(model)
    # sp_ub = SP_ub(model)
    # sp_lb = SP_lb(model)
    # w_sp, _ = spatial_branch_and_bound_maximize(
    #     sp_obj, sp_lb, sp_ub,
    #     (box_low, box_high),
    #     tolerance=tolerance
    # )
    # time_exact_sp = time.time() - start_time
    # x_exact_sp = model.SP(w_sp, solver='gurobi')[0]
    # x_exact_sp = np.array(x_exact_sp).flatten().tolist()


    # #############################################
    # ##### 2. solve OP
    # #############################################

    # # Create original problem solver
    # op = MPAssortOriginal(u, r, B, distr, C)
    # op_obj = OP_obj(op, distr.random_sample((10000, N+1)))

    # bf_optimizer = BruteForceOptimizer(N, C, num_cores=24)
    # start_time = time.time()
    # x_op, val_op = bf_optimizer.maximize(op_obj)
    # time_op = time.time() - start_time
    # x_op = np.array(x_op).flatten().tolist()



    # #############################################
    # ##### 3. compare op and sp
    # #############################################
    # pi_x_op = float(op_obj(x_op))
    # pi_x_sp = float(op_obj(x_exact_sp))

    # print(pi_x_op, pi_x_sp)
    # print(model._pi_hat(x_exact_sp), model._pi_hat(x_op))

    # # check range
    # print("--------------------------------- check range ---------------------------------")

    # print("======= start range =======")
    # for i in range(len(box_low)):
    #     print(box_low[i], box_high[i])
    # print("======= end range =======")

    # print(model._w_x(x_exact_sp), model._w_x(x_op))
