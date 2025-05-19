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

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder_path", type=str, required=True, help="Path to folder containing .pkl files")
    # args = parser.parse_args()
    # folder_path = args.folder_path

    if True:

        folder_path = r"results\0410_accuracy\scale_B\tol_0.0001"
        for filename in os.listdir(folder_path):
            print(filename)  
            if filename.startswith("raw") and filename.endswith(".pkl"):  
                file_path = os.path.join(folder_path, filename)

                # load data
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                # print(instances[0].keys())

                # break

                # 第二步：定义 key 的映射关系
                key_mapping = {
                    'time_exact_sp': 'time_sp_0.0001',
                    'x_exact_sp': 'x_sp_0.0001',
                    'pi_x_exact_sp': 'pi_x_sp_0.0001',
                    'time_exact_rsp': 'time_rsp_0.0001',
                    'x_exact_rsp': 'x_rsp_0.0001',
                    'pi_x_exact_rsp': 'pi_x_rsp_0.0001',
                    'time_clustered_sp': 'time_avg_sp_0.01',
                    'x_clustered_sp': 'x_avg_sp_0.01',
                    'pi_x_clustered_sp': 'pi_x_avg_sp_0.01',
                    'time_clustered_rsp': 'time_avg_rsp_0.01',
                    'x_clustered_rsp': 'x_avg_rsp_0.01',
                    'pi_x_clustered_rsp': 'pi_x_avg_rsp_0.01',
                    'time_op': 'time_opt',
                    'x_op': 'x_opt',
                    'pi_x_op': 'pi_x_opt'
                }

                # 第三步：处理每个字典，重命名 key 并去除值为 None 的项
                new_data = []
                for d in data:
                    new_dict = {}
                    for k, v in d.items():
                        if v is not None:  # 去除值为 None 的项
                            new_key = key_mapping.get(k, k)  # 重命名 key（如果有）
                            new_dict[new_key] = v
                    new_data.append(new_dict)
                
                with open(file_path, 'wb') as f:
                    pickle.dump(new_data, f)

    if False:
        folder_path = r"results\0410_accuracy\bai_et_al_setting"
        
        orginal_files = [
            'raw_dec_N_15_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_15_C_12_12_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_15_C_8_8_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_15_C_8_8_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_30_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_30_C_12_12_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_30_C_8_8_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_30_C_8_8_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_60_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_60_C_12_12_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_60_C_8_8_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_dec_N_60_C_8_8_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_15_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_15_C_12_12_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_15_C_8_8_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_15_C_8_8_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_30_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_30_C_12_12_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_30_C_8_8_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_30_C_8_8_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_60_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_60_C_12_12_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_60_C_8_8_B_1_2_3_distr_GumBel_tol_0.001_close_form.pkl',
            'raw_rand_N_60_C_8_8_B_1_2_distr_GumBel_tol_0.001_close_form.pkl',
        ]

        new_files = [
            'raw_dec_N_15_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_15_C_12_12_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_15_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_15_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_30_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_30_C_12_12_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_30_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_30_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_60_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_60_C_12_12_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_60_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_dec_N_60_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_15_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_15_C_12_12_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_15_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_15_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_30_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_30_C_12_12_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_30_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_30_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_60_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_60_C_12_12_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_60_C_8_8_B_1_2_3_distr_GumBel_tol_0.0001_close_form.pkl',
            'raw_rand_N_60_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form.pkl',
        ]

        added_keys = [
            'time_sp_0.0001',
            'x_sp_0.0001',
            'pi_x_sp_0.0001',
            'time_rsp_0.0001',
            'x_rsp_0.0001',
            'pi_x_rsp_0.0001',
        ]


        for i in range(len(orginal_files)):
            orginal_file = os.path.join(folder_path + r'\tol_0.001', orginal_files[i])
            new_file = os.path.join(folder_path + r'\tol_0.0001', new_files[i])

            with open(orginal_file, 'rb') as f:
                data1 = pickle.load(f)

            with open(new_file, 'rb') as f:
                data2 = pickle.load(f)

            new_data = []
            for j in range(len(data1)):
                d1 = data1[j]
                d2 = data2[j]
                for key in added_keys:
                    d1[key] = d2[key]
                if (i == 0) and (j == 0):
                    print(d1.keys())
                    print(d2.keys())
                new_data.append(d1)


            with open(os.path.join(folder_path, new_files[i]), 'wb') as f:
                pickle.dump(new_data, f)