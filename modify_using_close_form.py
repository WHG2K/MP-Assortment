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
from tqdm import tqdm


if __name__ == "__main__":

    # do two main things:
    # 1. if brute_force is not used, set pi_x_op, time_op, x_op to None
    # 2. evaluate all solutions using close form

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to folder containing .pkl files")
    args = parser.parse_args()
    folder_path = args.folder_path

    # folder_path = r"results\0410_accuracy\correct\test"

    df_list = []
    for filename in os.listdir(folder_path):
        print(filename)  
        if filename.startswith("raw") and filename.endswith(".pkl"):  
            file_path = os.path.join(folder_path, filename)

            # load data
            with open(file_path, 'rb') as f:
                instances = pickle.load(f)

            # transform
            df = pd.DataFrame(instances)

            # print(df.columns)

            # if "x_op" in df.columns:
            #     brute_force = True
            # else:
            #     brute_force = False

            # print(brute_force)

            x_list = ["x_exact_sp", "x_exact_rsp", "x_clustered_sp", "x_clustered_rsp", "x_mnl", "x_gr"]
            pi_list = ["pi_x_exact_sp", "pi_x_exact_rsp", "pi_x_clustered_sp", "pi_x_clustered_rsp", "pi_x_mnl", "pi_x_gr"]
            op_related_list = ["x_op", "pi_x_op", "time_op"]

            new_columns = dict()
            for c in pi_list:
                new_columns[c] = []
            for c in op_related_list:
                new_columns[c] = []

            for idx, row in tqdm(df.iterrows()):
                # print(idx, row["N"], row["C"])
                N = row["N"]
                C = row["C"]
                B = row["B"]
                distr = GumBel()
                u = row["u"]
                r = row["r"]
                model_op = MPAssortOriginal(u, r, B, distr, C)

                for idx, x in enumerate(x_list):
                    pi_name = pi_list[idx]
                    new_columns[pi_name].append(model_op._pi(row[x]))

                # print("evaluation finished.")

                if (N <= 20): # use brute force
                    # print("solving brute force")
                    op_obj = OP_obj(model_op)
                    bf_optimizer = BruteForceOptimizer(N, C, num_cores=24)
                    start_time = time.time()
                    x_op, val_op = bf_optimizer.maximize(op_obj)
                    new_columns['time_op'].append(time.time() - start_time)
                    new_columns['x_op'].append(np.array(x_op).reshape(-1).tolist())
                    new_columns['pi_x_op'].append(float(op_obj(x_op)))
                else:
                    new_columns['x_op'].append(None)
                    new_columns['pi_x_op'].append(None)
                    new_columns['time_op'].append(None)
                

            for c in pi_list:
                df[c] = new_columns[c]

            for c in op_related_list:
                df[c] = new_columns[c]

            new_file_path = file_path.replace(".pkl", "_close_form.pkl")

            with open(new_file_path, 'wb') as f:
                pickle.dump(df, f)