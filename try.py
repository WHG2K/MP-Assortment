import pandas as pd

# 示例结构
df = pd.DataFrame({
    'N': [10, 10, 20, 20],
    '|B|': [1, 1, 2, 2],
    'C': [0.1, 0.1, 0.2, 0.2],
    'randomness': [0.01, 0.01, 0.02, 0.02],
    'A': [5, 6, 7, 8],
    'B': [15, 16, 17, 18],
    'C_val': [25, 26, 27, 28],  # 避免和 'C' 列名冲突
})

result = df.groupby(['N', '|B|', 'C', 'randomness']).agg({
    'A': ['mean', ('p95', lambda x: x.quantile(0.95)), 'max'],
    'B': ['mean', ('p95', lambda x: x.quantile(0.95)), 'max'],
    'C_val': ['mean', ('p95', lambda x: x.quantile(0.95)), 'max'],
})

# # 给 lambda 命名一下列名更清晰
# result.columns = ['_'.join([col[0], col[1] if isinstance(col[1], str) else 'p95']) for col in result.columns]
# result = result.reset_index()

# print(result)


# 假设 df 已经存在
variables = ['A', 'B', 'C_val']

# 构建聚合字典
agg_dict = {
    var: [
        ('mean', 'mean'),
        ('p95', lambda x: x.quantile(0.95)),
        ('max', 'max')
    ] for var in variables
}

# 聚合计算
result = df.groupby(['N', '|B|', 'C', 'randomness']).agg(agg_dict)

# 整理列名（从多层列转为单层列）
result.columns = [f"{var}_{stat}" for var, stat in result.columns]
result = result.reset_index()

# LaTeX 输出
# latex_table = result.to_latex(index=False, float_format="%.2f")
# print(latex_table)

print(result)