import numpy as np
from collections import deque
    


# 分支定界算法
def branch_and_bound(f, f_lb, f_ub, box_low, box_high, tolerance=1e-3):
    # 初始化最优解和最佳下界
    best_solution = None
    best_objective = -np.inf
    lb = -np.inf   # 全局最优解的下界
    # ub = np.inf   # 全最优解的上界
    # lb = f_lb(box_low, box_high)
    # ub = f_ub(box_low, box_high)
    queue = deque([(box_low, box_high)])  # 队列存储子问题，开始时存储整个范围

    # 迭代分支定界
    while True:
        

        # 做二叉树的这一层，记录需要新加的东西，然后一次性加入
        layer_len = len(queue)
        ub = -np.inf   # 二叉树当前层的最优值上界

        # 对二叉树当前层的所有节点进行处理
        for _ in range(layer_len):
            box_low, box_high = queue.popleft()
            lb_current, x = f_lb(box_low, box_high)  # 计算当前子问题的下界
            ub_current = f_ub(box_low, box_high)  # 计算当前子问题的上界
            # 剪枝
            if ub_current < lb:
                continue
            # 更新lb, ub和全局最优解
            ub = max(ub, ub_current)
            if lb_current > lb:
                lb = lb_current
                best_solution = x
                best_objective = f(x)

            #### 分裂当前节点 ####
            # 选择当前区间中最长的维度进行分裂
            lengths = [box_high[i] - box_low[i] for i in range(len(box_low))]
            max_dim = np.argmax(lengths)  # 找到最长的维度
            
            # 将该维度分成两个子区间
            midpoint = (box_low[max_dim] + box_high[max_dim]) / 2
            left_box_low = np.copy(box_low)
            left_box_high = np.copy(box_high)
            left_box_high[max_dim] = midpoint

            right_box_low = np.copy(box_low)
            right_box_high = np.copy(box_high)
            right_box_low[max_dim] = midpoint

            queue.append((left_box_low, left_box_high))
            queue.append((right_box_low, right_box_high))

        print("ub=", ub, "lb=", lb)
        

        # 如果满足终止条件，即当前子问题的上界和下界之差小于容忍误差，输出当前中点
        if ub - lb < tolerance:
            # print("ub=", ub, "lb=", lb)
            # print("current_box_low=", current_box_low, "current_box_high=", current_box_high)
            # midpoint = np.array([(current_box_low[i] + current_box_high[i]) / 2 for i in range(3)])
            # return midpoint, f(midpoint)  # 输出当前分支的中点和其上界作为解
            return best_solution, f(best_solution)
    
