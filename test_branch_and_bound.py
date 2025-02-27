import numpy as np
from src.utils.lp_optimizers import LinearProgramSolver
from collections import deque
from src.algorithms.BB import branch_and_bound



'''
A test example for branch and bound
'''

# 定义目标函数
def test_obj(x):
    x = np.array(x, dtype=float).reshape(-1)
    return -np.sum(x ** 2) # 最大化问题，取负号

# 定义下界函数（最小可能值）
def test_lb(lower_bounds, upper_bounds):
    lower_bounds = np.array(lower_bounds, dtype=float).reshape(-1)
    upper_bounds = np.array(upper_bounds, dtype=float).reshape(-1)
    # 简单实现：使用上界点的目标函数值
    return -np.sum(lower_bounds ** 2), lower_bounds

# 定义上界函数（最大可能值）
def test_ub(lower_bounds, upper_bounds):
    lower_bounds = np.array(lower_bounds, dtype=float).reshape(-1)
    upper_bounds = np.array(upper_bounds, dtype=float).reshape(-1)
    # 简单实现：使用下界点的目标函数值
    a = np.zeros(len(lower_bounds), dtype=float)
    for i in range(len(lower_bounds)):
        if (lower_bounds[i] * upper_bounds[i]) > 0:
            a[i] = min(abs(lower_bounds[i]), abs(upper_bounds[i]))
        else:
            a[i] = 0
    return -np.sum(a ** 2)




'''
rsp objective functions
'''

class rsp_obj:
    def __init__(self, model):
        self.model = model

    def __call__(self, w):
        return self.model.RSP(w)
    
class rsp_ub:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):

        # Compute objective coefficients c
        c = self.model._probs_buying_surrogate(box_low) * self.model.r
        
        # Construct constraint matrix A
        A = np.vstack([
            self.model._probs_U_exceed_w(box_high),  # First |B| rows are P(u[j] + X > w[i])
            np.ones(self.N),   # Cardinality upper bound
            -np.ones(self.N)   # Cardinality lower bound
        ])
        
        # Compute RHS vector b
        b = np.concatenate([
            self.model._Bs,
            [self.model.C[1], -self.model.C[0]]
        ])

        lp_solver = LinearProgramSolver(c, A, b)
        upper_bound, _, status = lp_solver.maximize(c, A, b)
        if status != 'Optimal':
            raise ValueError(f"Failed to solve RSP upper bound: {status}")
        return upper_bound
    
class rsp_lb:
    def __init__(self, model):
        self.model = model

    def __call__(self, box_low, box_high):
        box_middle = (box_low + box_high) / 2
        rsp_box_low = self.model.RSP(box_low)
        rsp_box_high = self.model.RSP(box_high)
        rsp_box_middle = self.model.RSP(box_middle)

        return max(rsp_box_low, rsp_box_middle, rsp_box_high)
    


if __name__ == "__main__":

    # a = np.array([1, 2, 3])
    # print(a ** 2)
    
    box_low = np.array([-1, -1, -1], dtype=float)  # 目标函数下界
    box_high = np.array([2, 2, 2], dtype=float)  # 目标函数上界

    # 运行分支定界算法
    best_solution, best_objective = branch_and_bound(test_obj, test_lb, test_ub, box_low, box_high)

    print(f"最优解: {best_solution}")
    print(f"最优目标函数值: {best_objective}")

    # for _ in range(10):
    #     x = np.random.uniform(-1, 2, size=3)
    #     print(test_obj(x), test_lb(x-0.1, x+0.1), test_ub(x-0.1, x+0.1))

    print(test_lb([0.5, 0.5, 0.5], [2, 2, 2]))
    print(test_ub([0.5, 0.5, 0.5], [2, 2, 2]))
    