from src.algorithms.models import MPAssortSurrogate, MPAssortOriginal
from src.utils.brute_force import BruteForceOptimizer
import numpy as np
from src.utils.distributions import GumBel
from src.algorithms.sBB_functions_utils import RSP_obj, RSP_ub, RSP_lb, SP_obj, SP_ub, SP_lb, OP_obj
from dotenv import load_dotenv
from src.ptas.PTAS import AO_Instance, MP_MNL_PTAS
import os


if __name__ == "__main__":


    # np.random.seed(2025)
    # np.random.seed(0)

    load_dotenv(override=True)

    # check gurobi home and license
    gurobi_home = os.getenv("GUROBI_HOME")
    license_file = os.getenv("GRB_LICENSE_FILE")
    print(f"Gurobi home: {gurobi_home}")
    print(f"License path: {license_file}")

    N = 15
    u = np.random.normal(0, 1, N).reshape(-1).tolist()
    w = np.exp(u)
    w_max = np.max(w)
    r = (w_max - w).reshape(-1).tolist()

    # Generate basket size distribution
    basket_sizes = [1, 2, 3]
    probs = np.random.uniform(0, 1, len(basket_sizes))
    probs = probs / probs.sum()
    probs = probs.reshape(-1).tolist()
    B = dict(zip(basket_sizes, probs))

    C = (10, 10)

    # get parameters for ptas
    m = max(B.keys())
    lambda_ = [0.0] * (m + 1)  # index 0 unused, just set to 0.0
    for k in range(1, m + 1):
        lambda_[k] = B.get(k, 0.0)
    weights = np.exp(u).reshape(-1).tolist()

    # PTAS
    ao_instance = AO_Instance(N, m, lambda_, weights, r, C[0])
    # OP
    op = MPAssortOriginal(u, r, B, GumBel(), C)

    # compare revenue function
    x = np.random.choice([0, 1], size=N, p=[0.4, 0.6])
    S_x =[i for i, val in enumerate(x) if val > 0.99]
    print("ptas revenue", ao_instance.Compute_Rev(S_x))
    print("op revenue", op._pi(x))

    # compare choice probability
    i = np.random.choice(S_x)
    print("item", i)
    print("ptas choice probability", ao_instance.True_Choice_Prob(S_x, i))
    print("op choice probability", op.Get_Choice_Prob_MP_MNL(S_x, i))

    # compare optimal revenue
    best_rev_ptas = ao_instance.Get_Opt_Card()
    op_obj = OP_obj(op)
    brute_force = BruteForceOptimizer(N, C, num_cores=4)
    _, best_rev_op = brute_force.maximize(op_obj)
    print("best rev ptas", best_rev_ptas)
    print("best rev op", best_rev_op)


    # try ptas
    ptas_solver = MP_MNL_PTAS(ao_instance)
    best_rev_ptas = ptas_solver.solve(0.6)
    print("best rev ptas", best_rev_ptas)
    print("best assortment ptas", ptas_solver.best_S)
