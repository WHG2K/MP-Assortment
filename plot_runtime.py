import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set random seed
np.random.seed(42)

N_values = [10, 20, 40, 60, 100]
B_values = [1, 2, 3, 4, 6, 8, 10]
algorithms = ["alg1", "alg2"]

data = []
for B in B_values:
    for algo in algorithms:
        for N in N_values:
            runtimes = np.random.normal(loc=N * (1.2 if algo == "alg1" else 0.5) * (1 + B * 0.02), 
                                        scale=N * 0.2, size=15)  # 生成不同 B 的 runtime
            for runtime in runtimes:
                data.append([algo, B, N, runtime])

# create DataFrame
df = pd.DataFrame(data, columns=["algorithm", "B", "N", "runtime"])

# print(df)


# 选择要绘图的 B 值
B_fixed = 4  # 你可以修改这个值，比如 20

# 过滤 DataFrame
df_filtered = df[df["B"] == B_fixed]

# 画图
plt.figure(figsize=(8,6))

# sns.lineplot(
#     x="N", y="runtime", hue="algorithm", style="B", data=df_filtered, 
#     marker="o", linewidth=2, errorbar=('ci', 95)  # Q1-Q3 作为误差范围
# )

sns.lineplot(
    x="N", y="runtime", hue="algorithm", data=df_filtered, 
    marker="o", linewidth=2, errorbar=('ci', 95)  # Q1-Q3 作为误差范围
)

plt.xlabel("N")
plt.ylabel("Runtime")
plt.title("Runtime vs. N for Different B Values with Q1-Q3")
plt.legend(title="Algorithm")

# 设置更适合论文的字体和格式
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 增加合适的边距
plt.tight_layout()

# 保存为 PDF，设置 DPI 为 300
plt.savefig("runtime_vs_N_B_10.pdf", format="pdf", dpi=300)

plt.show()
