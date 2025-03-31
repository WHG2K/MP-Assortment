import pandas as pd
import numpy as np

# 参数设置
num_simulations = 10000000  # 模拟次数

# 生成模拟数据
data = {
    # A 投掷 3 个 6 面骰子
    'A1': np.random.randint(1, 9, size=num_simulations),
    'A2': np.random.randint(1, 9, size=num_simulations),
    'A3': np.random.randint(1, 9, size=num_simulations),
    # B 投掷 1 个 20 面骰子
    'B': np.random.randint(1, 27, size=num_simulations),
    # C 投掷 1 个 20 面骰子
    'C': np.random.randint(1, 27, size=num_simulations),
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 计算 A 的总和
df['A_sum'] = df['A1'] + df['A2'] + df['A3']

# 比较 A、B 和 C 的点数
df['A_wins'] = (df['A_sum'] > df['B']) & (df['A_sum'] > df['C'])
df['B_wins'] = (df['B'] > df['A_sum']) & (df['B'] > df['C'])
df['C_wins'] = (df['C'] > df['A_sum']) & (df['C'] > df['B'])
df['A_win_B'] = (df['A_sum'] > df['B'])
df["A=B>C"] = (df['A_sum'] == df['B']) & (df['B'] > df['C'])

# 计算概率
prob_a_wins = df['A_wins'].mean()
prob_b_wins = df['B_wins'].mean()
prob_c_wins = df['C_wins'].mean()
prob_a_win_b = df['A_win_B'].mean()
prob_a_equal_b_and_c = df['A=B>C'].mean()
# 输出结果
print(f"A 胜利的概率: {prob_a_wins:.4f}")
print(f"B 胜利的概率: {prob_b_wins:.4f}")
print(f"C 胜利的概率: {prob_c_wins:.4f}")
print(f"A比B大的概率: {prob_a_win_b:.4f}")
print(f"A=B>C的概率: {prob_a_equal_b_and_c:.4f}")
# print(prob_a_wins + prob_b_wins + prob_c_wins)