import numpy as np
import pandas as pd

# 设置模拟参数
n = 3  # 样本数，可以根据需要调整
num_samples = 1000000  # 模拟样本数量，越大越精确

# 生成模拟数据
# 生成 n 个均匀分布的随机变量 X1, X2, ..., Xn
X = np.random.uniform(0, 1, (num_samples, n))

# 计算 Y = min(X1, X2, ..., Xn) 和 Z = max(X1, X2, ..., Xn)
Y = np.min(X, axis=1)
Z = np.max(X, axis=1)

# 创建 DataFrame 来加速计算
df = pd.DataFrame({'Y': Y, 'Z': Z})

# 计算相关性
correlation = df.corr().iloc[0, 1]

# 输出结果
print(f'模拟的 Y 和 Z 的相关系数: {correlation}')