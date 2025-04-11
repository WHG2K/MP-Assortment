import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

def get_fractional_indices_data():

    # load all pkl files
    folder_path = r"results\0408_sparsity"  # 替换为你的文件夹路径
    df_list = []
    for filename in os.listdir(folder_path):
        if filename.startswith("raw") and filename.endswith(".pkl"):  
            file_path = os.path.join(folder_path, filename)

            # load data
            with open(file_path, 'rb') as f:
                instances = pickle.load(f)

            # transform
            df = pd.DataFrame(instances)
            df["|B|"] = df["B"].apply(len)
            # df_new = pd.melt(df, id_vars=["N", "|B|"], value_vars=["time_exact_sp", "time_exact_rsp"],
            #                 var_name="alg", value_name="runtime")
            # df_new["alg"] = df_new["alg"].replace({"time_exact_sp": "SP", "time_exact_rsp": "RSP"})
            df["count_frac"] = df["x_bar_exact_rsp"].apply(lambda x: sum(0.001 < a < 0.999 for a in x))

            df_list.append(df[["N", "|B|", "count_frac"]])

    # merge all dataframes
    df = pd.concat(df_list, axis=0)

    return df


def plot_fractional_indices(df):

    folder_path = r"results\0408_sparsity"

    os.makedirs(folder_path + r"\histograms", exist_ok=True)
    updated_folder_path = folder_path + r"\histograms"

    for (n, b), group in df.groupby(['N', '|B|']):
        plt.figure(figsize=(6, 4))
        # sns.histplot(group['count_frac'], discrete=True)
        sns.countplot(x='count_frac', data=group, color='skyblue')
        # plt.title(f'Histogram of count_frac (N={n}, |B|={b})')
        plt.xlabel('$|\Delta|$')
        plt.ylabel('Frequency')
        plt.tight_layout()

        filename = updated_folder_path  + f"\hist_count_frac_N_{n}_Bsize_{b}.png"
        plt.savefig(filename)
        plt.close()



if __name__ == "__main__":

    df = get_fractional_indices_data()
    # df.to_excel("check_fractional_indices.xlsx", index=False)
    # plot_runtime(df)
    plot_fractional_indices(df)
