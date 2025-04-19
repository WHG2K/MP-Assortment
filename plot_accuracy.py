import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

def simplify_tuple(t):
    if isinstance(t, tuple) and len(t) == 2:
        return t[0] if t[0] == t[1] else t
    return t  # 万一有异常值，保持原样

def get_accuracy_data(brute_force=False):

    # load all pkl files
    folder_path = r"results\0410_accuracy\scale_B\tol_0.0001"
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
            # df["|B|"] = df["B"].apply(len)
            df["B"] = df["B"].apply(lambda d: tuple(sorted(d.keys())))
            df["C"] = df["C"].apply(simplify_tuple)
            # df_new = pd.melt(df, id_vars=["N", "|B|"], value_vars=["time_exact_sp", "time_exact_rsp"],
            #                 var_name="alg", value_name="runtime")
            # df_new["alg"] = df_new["alg"].replace({"time_exact_sp": "SP", "time_exact_rsp": "RSP"})
            if brute_force:
                df_list.append(df[["N", "C", "B", "randomness", "pi_x_op", "pi_x_exact_sp", "pi_x_exact_rsp", "pi_x_clustered_sp", "pi_x_clustered_rsp", "pi_x_mnl", "pi_x_gr"]])
            else:
                df_list.append(df[["N", "C", "B", "randomness", "pi_x_exact_sp", "pi_x_exact_rsp", "pi_x_clustered_sp", "pi_x_clustered_rsp", "pi_x_mnl", "pi_x_gr"]])

    # merge all dataframes
    df = pd.concat(df_list, axis=0)

    return df


def table_accuracy(df, variables=["gap_sp", "gap_rsp", "gap_mnl", "gap_gr"]):

    agg_dict = {
        var: [
            ('mean', 'mean'),
            ('p95', lambda x: x.quantile(0.95)),
            ('max', 'max')
        ] for var in variables
    }

    # calculate gaps
    df["gap_sp"] = 1 - df["pi_x_exact_sp"] / df["pi_x_op"]
    df["gap_rsp"] = 1 - df["pi_x_exact_rsp"] / df["pi_x_op"]
    df["gap_mnl"] = 1 - df["pi_x_mnl"] / df["pi_x_op"]
    df["gap_gr"] = 1 - df["pi_x_gr"] / df["pi_x_op"]

    # 聚合计算
    result = df.groupby(['N', '|B|', 'C', 'randomness']).agg(agg_dict)

    stats_order = ['mean', 'p95', 'max']

    # 整理列名（从多层列转为单层列）
    # result.columns = [f"{var}_{stat}" for var, stat in result.columns]
    new_columns = [(var, stat) for stat in stats_order for var in variables]
    result = result[new_columns]
    # print(result.columns)
    result = result.reset_index()

    for _, row in result.iterrows():
        line = ' & '.join([f"{100*row[col]:.2f}\%" if isinstance(row[col], float) else str(row[col]) for col in result.columns])
        print(line + r' \\')

    return result

def table_accuracy_relative(df, variables=["sp2mnl", "rsp2mnl", "sp2gr", "rsp2gr"]):

    agg_dict = {
        var: [
            ('mean', 'mean'),
            ('p95', lambda x: x.quantile(0.95)),
            ('max', 'max')
        ] for var in variables
    }

    # calculate gaps
    # df["gap_sp"] = 1 - df["pi_x_exact_sp"] / df["pi_x_op"]
    # df["gap_rsp"] = 1 - df["pi_x_exact_rsp"] / df["pi_x_op"]
    # df["gap_mnl"] = 1 - df["pi_x_mnl"] / df["pi_x_op"]
    # df["gap_gr"] = 1 - df["pi_x_gr"] / df["pi_x_op"]
    df["sp2mnl"] = df["pi_x_exact_sp"] / df["pi_x_mnl"] - 1
    df["rsp2mnl"] = df["pi_x_exact_rsp"] / df["pi_x_mnl"] - 1
    df["sp2gr"] = df["pi_x_exact_sp"] / df["pi_x_gr"] - 1
    df["rsp2gr"] = df["pi_x_exact_rsp"] / df["pi_x_gr"] - 1

    # 聚合计算
    # result = df.groupby(['N', '|B|', 'C', 'randomness']).agg(agg_dict)
    result = df.groupby(['N', 'B', 'C', 'randomness']).agg(agg_dict)

    stats_order = ['mean', 'p95', 'max']

    # 整理列名（从多层列转为单层列）
    # result.columns = [f"{var}_{stat}" for var, stat in result.columns]
    new_columns = [(var, stat) for stat in stats_order for var in variables]
    result = result[new_columns]
    # print(result.columns)
    result = result.reset_index()

    for _, row in result.iterrows():
        line = ' & '.join([f"{100*row[col]:.2f}\%" if isinstance(row[col], float) else str(row[col]) for col in result.columns])
        print(line + r' \\')

    return result



if __name__ == "__main__":

    df = get_accuracy_data(brute_force=False)
    # df.to_excel("accuracy_data.xlsx", index=False)
    # result = table_accuracy(df)
    result = table_accuracy_relative(df)
    # print(result)
    # print(df.head())
