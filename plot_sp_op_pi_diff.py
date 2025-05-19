import argparse
import numpy as np
from src.utils.distributions import GumBel
import pickle
from tqdm import tqdm
import time
from src.algorithms.models import MPAssortSurrogate, MPAssortOriginal
from src.ptas.PTAS import AO_Instance, MP_MNL_PTAS
from src.algorithms.sBB import spatial_branch_and_bound_maximize
from src.utils.brute_force import BruteForceOptimizer
from src.utils.greedy import GreedyOptimizer
from src.algorithms.models import MNL
from src.algorithms.sBB_functions_utils import RSP_obj, RSP_ub, RSP_lb, SP_obj, SP_ub, SP_lb, OP_obj
from dotenv import load_dotenv
import os
import pandas as pd

# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_instances(N=15, C=(8,8), B=[3], randomness='rand', distr="GumBel", random_seed=2025, n_instances=100):
    """Generate multiple problem instances

    Args:
        N: Number of products
        C: Tuple of (lower_bound, upper_bound) for cardinality constraint
        distr: Distribution object (default: GumBel)
        random_seed: Random seed for reproducibility
        n_instances: Number of instances to generate

    Returns:
        list: List of instance dictionaries
    """
    # Set global random seed
    np.random.seed(random_seed)

    instances = []
    for i in range(n_instances):
        # Generate utilities and revenues
        u = np.random.normal(0, 1, N).reshape(-1).tolist()
        w = np.exp(u)
        w_max = np.max(w)
        r = (w_max - w).reshape(-1).tolist()

        # Generate basket size distribution
        basket_sizes = B
        probs = np.random.uniform(0, 1, len(basket_sizes))
        if randomness == 'rand':
            pass
        elif randomness == 'dec':
            probs = np.sort(probs)[::-1]
        else:
            raise ValueError(f"Unsupported randomness: {randomness}")
        probs = probs / probs.sum()
        probs = probs.reshape(-1).tolist()
        B = dict(zip(basket_sizes, probs))

        # Store instance data
        instance = {
            'instance_id': i,
            'N': N,
            'C': C,
            'randomness': randomness,
            'distr': distr,
            'u': u,
            'r': r,
            'B': B
        }
        instances.append(instance)

    return instances

def save_instances(instances, file_name):
    """Save instances to pickle file

    Args:
        instances: List of instance dictionaries
        file_name: Name of pickle file to save
    """
    with open(file_name, 'wb') as f:
        pickle.dump(instances, f)



if __name__ == "__main__":
        
    if False:

        file_name = "section_4_instances.pkl"
        full_instances = []

        # generate instances
        print("Generating problem instances...")
        for b in [1,2,3,4,5]:
            for N in [10,20,30,40,50,60,70,80,90,100]:
                full_instances.extend(generate_instances(N=N, C=(8,8), B=[b], randomness='dec', distr="GumBel", random_seed=2025, n_instances=10))

        save_instances(full_instances, file_name)


        # compute
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)

        for idx, inst in enumerate(tqdm(instances)):

            # initialize model and get box constraints
            if inst['distr'] == "GumBel":
                distr = GumBel()
            else:
                raise ValueError(f"Unsupported distribution: {inst['distr']}")
            

            model_sp = MPAssortSurrogate(u=inst['u'], r=inst['r'], B=inst['B'], 
                                    distr=distr, C=inst['C'])
            model_op = MPAssortOriginal(u=inst['u'], r=inst['r'], B=inst['B'], 
                                    distr=distr, C=inst['C'])
            
            N = inst['N']
            x = np.ones(N, dtype=int)
            # S_x = [i for i in range(N) if x[i] == 1]
            # probs_op = [model_op.Get_Choice_Prob_MP_MNL(S_x, i) for i in range(N)]
            # w_x = model_sp._w_x(x)
            # probs_sp = model_sp._compute_LP_parameters(w_x)[0]
            pi_hat_x = model_sp._pi_hat(x)
            pi_x = model_op._pi(x)
            inst['pi_hat_x'] = pi_hat_x
            inst['pi_x'] = pi_x
            inst['pi_rel_diff'] = (pi_hat_x - pi_x) / pi_x

            # # diff = np.sum(np.abs(np.array(probs_op) - np.array(probs_sp)))
            # diff = np.abs(np.sum(np.array(probs_op) - np.sum(probs_sp)))
            # inst['prob_op'] = probs_op
            # inst['prob_sp'] = probs_sp
            # inst['prob_diff'] = probs_op - probs_sp
            # inst['prob_rel_diff'] = diff / np.sum(np.array(probs_op))

        # Save updated instances
        with open(file_name, 'wb') as f:
            pickle.dump(full_instances, f)
        print(f"\nUpdated instances saved to {file_name}\n")

        df = pd.DataFrame(instances)
        df.to_excel("section_4_instances.xlsx", index=False)


    # heat map
    if True:

        # 读取 Excel 文件
        df = pd.read_excel("section_4_instances.xlsx")

        # 提取 B 值（你已确认安全）
        df['B_value'] = df['B'].apply(lambda b: list(eval(b).keys())[0])

        # 计算每组 (N, B) 下的平均相对误差
        heatmap_data = df.groupby(['N', 'B_value'])['pi_rel_diff'].mean().unstack()
        heatmap_data = heatmap_data.sort_index(ascending=False)  # 反转 N 的顺序

        # 绘制热力图
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Average Relative Gap'}, annot_kws={"size": 12})

        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=12)
        colorbar.set_label('Average Relative Gap', fontsize=14)

        # plt.title("Heatmap of Mean Relative Gap by N and B", fontsize=14)
        plt.xlabel(r"$B$", fontsize=14)
        plt.ylabel(r"$N$", fontsize=14)
        plt.tight_layout()
        plt.savefig("sp_op_diff_heatmap.pdf", dpi=300, format='pdf')
        plt.show()



    
    # line plot
    if True:
        # 读取数据
        df = pd.read_excel("section_4_instances.xlsx")

        # 提取 B 值（安全使用 eval）
        df['B_value'] = df['B'].apply(lambda b: list(eval(b).keys())[0])

        # 计算每组 (N, B) 的平均相对误差
        line_data = df.groupby(['N', 'B_value'])['pi_rel_diff'].mean().reset_index()

        # 绘图：x轴为 N，线条按 B 分组
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=line_data, x='N', y='pi_rel_diff', hue='B_value', marker='o', palette='tab10')



        plt.ylim(bottom=0)  # 设置 y 轴从 0 开始
        xticks = sorted(list(set(line_data['N']).union({10})))  # 确保 N=10 在刻度中
        plt.xticks(xticks, fontsize=12)

        # 设置标题和标签
        # plt.title(r"Mean Relative Gap $|1 - \hat{\pi}(e)/\pi(e)|$ vs. $N$, grouped by $B$", fontsize=14)
        # plt.title(r"Mean Relative Gap: $|1 - \hat{\pi}(e)/\pi(e)|$ vs. $N$ (Grouped by $B$)", fontsize=14)
        plt.xlabel(r"$N$", fontsize=14)
        plt.ylabel("Average Relative Gap", fontsize=14)
        plt.legend(title=r"$b$", fontsize=12, title_fontsize=14)

        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.savefig("sp_op_diff_lineplot.pdf", dpi=300, format='pdf')
        plt.show()
