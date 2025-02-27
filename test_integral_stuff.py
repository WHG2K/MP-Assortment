import numpy as np
from scipy.special import softmax
from src.utils.distributions import GumBel
from src.algorithms.solvers import MPAssortSurrogate
import time

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 问题参数
    N = 25
    C = (3, 7)
    
    # 生成随机实例
    u = np.random.normal(0, 1, N)
    u_max = np.max(u)
    r = np.exp(u_max) - np.exp(u)
    
    # 生成购物篮大小分布
    basket_sizes = [1, 2, 3]
    probs = np.random.normal(0, 1, len(basket_sizes))
    probs = softmax(probs)
    B = dict(zip(basket_sizes, probs))

    # 设置分布
    distr = GumBel()
    
    # 创建求解器实例
    sp = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
    
    # 生成一个随机的w向量进行测试
    w = np.random.rand(len(B))
    
    # 调用_probs_buying_surrogate方法
    time_start = time.time()
    probs = sp._probs_buying_surrogate(w)
    time_end = time.time()
    print(f"时间消耗: {time_end - time_start} 秒")
    
    time_start = time.time()
    probs_benchmark = distr._compute_c_vector(u, w, sp._B_probs)
    time_end = time.time()
    print(f"时间消耗: {time_end - time_start} 秒")

    print(f"diff: {np.abs(probs - probs_benchmark)}")
