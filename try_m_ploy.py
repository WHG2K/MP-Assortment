import numpy as np

def is_valid_polygon(edges):
    """检查边长是否能组成一个有效的多边形"""
    total_length = sum(edges)
    for edge in edges:
        if total_length - edge <= edge:
            return False
    return True

def simulate_polygon_probability(m, num_trials):
    """模拟 m 边多边形的概率"""
    valid_polygon_count = 0
    for _ in range(num_trials):
        # 随机生成 m 个边长
        edges = np.random.uniform(0, 1, m)
        if is_valid_polygon(edges):
            valid_polygon_count += 1
    return valid_polygon_count / num_trials

# 设置参数
num_trials = 1000000  # 模拟的试验次数
m = 4  # 设置多边形的边数

# 模拟 m 边多边形的概率
polygon_probability = simulate_polygon_probability(m, num_trials)
print(f"模拟的 {m} 边多边形概率: {polygon_probability}")