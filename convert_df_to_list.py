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

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to folder containing .pkl files")
    args = parser.parse_args()
    folder_path = args.folder_path

    # folder_path = r"results\0410_accuracy\correct\test"
    for filename in os.listdir(folder_path):
        print(filename)  
        if filename.startswith("raw") and filename.endswith(".pkl"):  
            file_path = os.path.join(folder_path, filename)

            # load data
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                instances = df.to_dict(orient="records")
            
            with open(file_path, 'wb') as f:
                pickle.dump(instances, f)

            # # load data
            # with open(file_path, 'rb') as f:
            #     df = pickle.load(f)
            #     print(df)

            