import numpy as np
from collections import deque
    


# Branch and Bound Algorithm
def branch_and_bound(f, f_lb, f_ub, box_low, box_high, tolerance=1e-3):
    # Initialize optimal solution and best lower bound
    best_solution = None
    best_objective = -np.inf
    lb = -np.inf   # Global lower bound of optimal solution
    # ub = np.inf   # Global upper bound of optimal solution
    # lb = f_lb(box_low, box_high)
    # ub = f_ub(box_low, box_high)
    queue = deque([(box_low, box_high)])  # Queue stores subproblems, initially stores the entire range
    layer = 0

    # Branch and bound iteration
    while True:
        
        # Process current layer of binary tree, record new nodes to be added
        layer_len = len(queue)
        ub = -np.inf   # Upper bound of current tree layer

        # Process all nodes in current layer
        for _ in range(layer_len):
            box_low, box_high = queue.popleft()
            lb_current, x = f_lb(box_low, box_high)  # Calculate lower bound of current subproblem
            ub_current = f_ub(box_low, box_high)  # Calculate upper bound of current subproblem
            # Pruning
            if ub_current < lb:
                continue
            # Update lb, ub and global optimal solution
            ub = max(ub, ub_current)
            if lb_current > lb:
                lb = lb_current
                best_solution = x
                best_objective = f(x)

            #### Split current node ####
            # Choose the longest dimension in current interval for splitting
            lengths = [box_high[i] - box_low[i] for i in range(len(box_low))]
            max_dim = np.argmax(lengths)  # Find the longest dimension
            
            # Split this dimension into two subintervals
            midpoint = (box_low[max_dim] + box_high[max_dim]) / 2
            left_box_low = np.copy(box_low)
            left_box_high = np.copy(box_high)
            left_box_high[max_dim] = midpoint

            right_box_low = np.copy(box_low)
            right_box_high = np.copy(box_high)
            right_box_low[max_dim] = midpoint

            queue.append((left_box_low, left_box_high))
            queue.append((right_box_low, right_box_high))

        # print(f"Layer={layer:2d}, ub={ub:.4f}, lb={lb:.4f}")
        layer += 1
        
        if ub - lb < tolerance:
            return best_solution, f(best_solution)
    
