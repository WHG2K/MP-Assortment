import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

def get_runtime_data():

    # load all pkl files
    folder_path = r"results\0402_runtime"  # 替换为你的文件夹路径
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
            df_new = pd.melt(df, id_vars=["N", "|B|"], value_vars=["time_exact_sp", "time_exact_rsp"],
                            var_name="alg", value_name="runtime")
            df_new["alg"] = df_new["alg"].replace({"time_exact_sp": "SP", "time_exact_rsp": "RSP"})

            df_list.append(df_new)

    # merge all dataframes
    df = pd.concat(df_list, axis=0)

    return df


# def plot_runtime(df):

#     folder_path = r"results\0402_runtime"
#     os.makedirs(folder_path + r"\runtime_plots", exist_ok=True)
#     updated_folder_path = folder_path + r"\runtime_plots"

#     for B_fixed in df["|B|"].unique():

#         # filter dataframe
#         df_filtered = df[df["|B|"] == B_fixed]

#         # plot
#         plt.figure(figsize=(8,6))
#         sns.lineplot(
#             x="N", y="runtime", hue="alg", data=df_filtered, 
#             marker="o", linewidth=2, errorbar=('ci', 95)  # Q1-Q3 作为误差范围
#         )
#         plt.xlabel("N")
#         plt.ylabel("Runtime (s)")
#         # plt.title("Runtime vs. N for Different B Values with Q1-Q3")
#         # plt.legend(title="Algorithm")
#         plt.legend()

#         # decorate
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.tight_layout()

#         # save
#         plt.savefig(updated_folder_path + "\\" + f"runtime_Bsize_{B_fixed}.pdf", format="pdf", dpi=300)

#         # plt.show()


def plot_runtime_boxplots(df):

    folder_path = r"results\0402_runtime"
    os.makedirs(folder_path + r"\runtime_boxplots", exist_ok=True)
    updated_folder_path = folder_path + r"\runtime_boxplots"

    for B_fixed in sorted(df["|B|"].unique()):

        palette = {"SP": "#1f77b4", "RSP": "#ff7f0e"}

        # filter dataframe
        df_filtered = df[df["|B|"] == B_fixed]

        plt.figure(figsize=(10, 6))

        # 画 boxplot，按照算法分颜色，并排显示
        sns.boxplot(
            x="N", y="runtime", hue="alg", data=df_filtered,
            showfliers=False, dodge=True, palette=palette, width=0.6
        )

        # # 添加中位数连接线
        # for alg in df_filtered["alg"].unique():
        #     medians = df_filtered[df_filtered["alg"] == alg].groupby("N")["runtime"].median().reset_index()
        #     plt.plot(medians["N"], medians["runtime"], marker="o", label=f"{alg} median", color=palette[alg])

        # # 遍历算法
        # for alg in df_filtered["alg"].unique():
        #     df_alg = df_filtered[df_filtered["alg"] == alg]

        #     # 绘制 boxplot
        #     sns.boxplot(
        #         x="N", y="runtime", data=df_alg, 
        #         color="lightgray",  # 所有算法用灰色底（也可以改成 palette）
        #         fliersize=3, width=0.5,
        #         showfliers=False
        #     )

            # # 算每个 N 下的中位数
            # medians = df_alg.groupby("N")["runtime"].median().reset_index()

            # # 画趋势线
            # plt.plot(medians["N"], medians["runtime"], marker="o", label=alg)

        plt.xlabel("N")
        plt.ylabel("Runtime (s)")
        # plt.title(f"Runtime vs. N (|B| = {B_fixed})")
        plt.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # save
        plt.savefig(updated_folder_path + "\\" + f"runtime_Bsize_{B_fixed}_boxplot.pdf", format="pdf", dpi=300)



if __name__ == "__main__":

    df = get_runtime_data()
    plot_runtime_boxplots(df)
