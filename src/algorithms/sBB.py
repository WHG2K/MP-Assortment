import numpy as np
from collections import deque
# from multiprocessing import Pool, cpu_count



# def spatial_branch_and_bound(f, f_lb, f_ub, R, tolerance=0.05):
#     """Branch and Bound algorithm with parallel processing
    
#     Args:
#         f: objective function
#         f_lb: lower bound function
#         f_ub: upper bound function
#         box_low: lower bounds of initial box
#         box_high: upper bounds of initial box
#         tolerance: convergence tolerance
#         min_box_size: minimum box size for splitting
#         num_workers: number of parallel workers (default: number of CPU cores - 1)
#     """

#     regions = [{"region": R, "lower_bound": -np.inf}]  # Step 1: Initialize list of regions
#     U = np.inf  # Best objective value found so far
#     x_star = None  # Best solution found so far
    
#     iter = 0
#     while regions:
#         # Step 2: Choose the region with the lowest lower bound
#         regions.sort(key=lambda r: r["lower_bound"])
#         node = regions.pop(0)
#         R = node["region"]
        
#         # Step 3: Compute lower bound by solving a convex relaxation
#         # if relaxation_solver:
#         #     l, x_relaxed = relaxation_solver(f, bounds)
#         # else:
#         #     l, x_relaxed = naive_relaxation(f, bounds)
#         l = f_lb(R)
        
#         if l > U:
#             continue  # Prune this region
        
#         # Step 4: Compute upper bound by solving original problem locally
#         # if local_solver:
#         #     u, x_local = local_solver(f, bounds)
#         # else:
#         #     u, x_local = local_optimization(f, bounds)
#         u, x_local = f_ub(R)
        
#         if u < U:
#             U = u
#             x_star = x_local
#             # Step 5: Pruning
#             regions = [r for r in regions if r["lower_bound"] > U]

#         print(f"iteration {iter:03d}, U: {U:0.4f}, l: {l:0.4f}")
        
#         # Step 6: Check convergence
#         if u - l <= tolerance:
#             print(f"Converged at {x_star} with value {U}")
#             return x_star, f(x_star)  # Accept this region's solution
        
#         # Step 7: Branching
#         # sub_regions = branch_region(bounds)
#         lengths = [R[0][i] - R[1][i] for i in range(len(R[0]))]
#         max_dim = np.argmax(lengths)
        
#         midpoint = (R[0][max_dim] + R[1][max_dim]) / 2
        
#         left_box_low = np.copy(R[0])
#         left_box_high = np.copy(R[1])
#         left_box_high[max_dim] = midpoint
#         R_left = (left_box_low, left_box_high)

#         right_box_low = np.copy(R[0])
#         right_box_high = np.copy(R[1])
#         right_box_low[max_dim] = midpoint
#         R_right = (right_box_low, right_box_high)
        
#         sub_regions = [R_left, R_right]

#         for sub_R in sub_regions:
#             regions.append({"region": sub_R, "lower_bound": l})  # Assign initial lower bound

#         iter += 1
    
#     print("List becomes empty")
#     return x_star, f(x_star)



def spatial_branch_and_bound_maximize(f, f_lb, f_ub, R, tolerance=0.05):
    """Branch and Bound algorithm with parallel processing
    
    Args:
        f: objective function
        f_lb: lower bound function
        f_ub: upper bound function
        box_low: lower bounds of initial box
        box_high: upper bounds of initial box
        tolerance: convergence tolerance
        min_box_size: minimum box size for splitting
        num_workers: number of parallel workers (default: number of CPU cores - 1)
    """

    regions = [{"region": R, "upper_bound": np.inf}]  # Step 1: Initialize list of regions
    L = -np.inf  # Best objective value found so far
    x_star = None  # Best solution found so far
    
    iter = 0
    # while regions and (iter < 20):
    while regions:
        # Step 2: Choose the region with the lowest lower bound
        regions.sort(key=lambda r:-r["upper_bound"])
        # print([r["upper_bound"] for r in regions])
        node = regions.pop(0)
        R = node["region"]
        
        # Step 3: Compute lower bound by solving a convex relaxation
        # if relaxation_solver:
        #     l, x_relaxed = relaxation_solver(f, bounds)
        # else:
        #     l, x_relaxed = naive_relaxation(f, bounds)
        # l = f_lb(R)
        u = f_ub(R)
        
        if u < L:
            continue  # Prune this region
        
        # Step 4: Compute upper bound by solving original problem locally
        # if local_solver:
        #     u, x_local = local_solver(f, bounds)
        # else:
        #     u, x_local = local_optimization(f, bounds)
        l, x_local = f_lb(R)
        
        if l > L:
            L = l
            x_star = x_local
            # Step 5: Pruning
            regions = [r for r in regions if r["upper_bound"] < L]



        # print(f"iteration {iter:5d}, u: {u:0.4f}, L: {L:0.4f}, N_nodes: {len(regions)}")

        # Step 6: Check convergence
        if u - L <= tolerance * u:
            # print(f"Converged at {x_star} with value {f(x_star)}")
            return x_star, f(x_star)  # Accept this region's solution
        
        # Step 7: Branching
        lengths = [R[1][i] - R[0][i] for i in range(len(R[0]))]
        max_dim = np.argmax(lengths)
        
        midpoint = (R[0][max_dim] + R[1][max_dim]) / 2
        
        left_box_low = np.copy(R[0])
        left_box_high = np.copy(R[1])
        left_box_high[max_dim] = midpoint
        R_left = (left_box_low, left_box_high)

        right_box_low = np.copy(R[0])
        right_box_high = np.copy(R[1])
        right_box_low[max_dim] = midpoint
        R_right = (right_box_low, right_box_high)
        
        sub_regions = [R_left, R_right]

        for sub_R in sub_regions:
            regions.append({"region": sub_R, "upper_bound": u})  # Assign initial lower bound

        iter += 1
    
    print("List becomes empty")
    return x_star, f(x_star)