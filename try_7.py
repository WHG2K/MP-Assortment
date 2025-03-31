import numpy as np

# 模拟的试验次数
num_trials = 1000000

# 从1到8的离散均匀分布生成3个独立的随机变量
random_variables = np.random.randint(1, 9, (num_trials, 3))

# 计算它们的和
sums = np.sum(random_variables, axis=1)

# 计算均值和二阶矩
mean = np.mean(sums)
second_moment = np.mean(sums**2)

# 计算方差

print(f"和的均值: {mean}")
print(f"和的二阶矩: {second_moment}")