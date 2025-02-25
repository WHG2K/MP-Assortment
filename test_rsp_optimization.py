import numpy as np
from scipy.special import softmax
from src.utils.distributions import GumBel
from src.algorithms.solvers import MPAssortSurrogate, MPAssortOriginal
from src.utils.nonconvex_optimizer import NonconvexOptimizer
from src.utils.brute_force import BruteForceOptimizer
import time

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Problem parameters
    N = 12  # Number of products
    C = (3, 7)  # Cardinality constraints
    
    # Generate random problem instance
    u = np.random.normal(0, 1, N)
    u_max = np.max(u)
    r = np.exp(u_max) - np.exp(u)  # Modified revenue calculation
    
    # Generate basket size distribution
    basket_sizes = [1, 2, 3]
    probs = np.random.normal(0, 1, len(basket_sizes))
    probs = softmax(probs)
    B = dict(zip(basket_sizes, probs))

    # distribution setting
    distr = GumBel()
    
    # Create solver instance
    sp = MPAssortSurrogate(u=u, r=r, B=B, distr=distr, C=C)
    
    # Get box constraints and create optimizer
    bounds = sp._get_box_constraints()
    print("\n=== Box Constraints ===")
    for i, b in enumerate(B.keys()):
        print(f"Range of w[{i}]: [{bounds[i][0]:.4f}, {bounds[i][1]:.4f}]")
    print()
    
    # Test both solvers
    print("\n=== Bayesian Optimization ===")
    optimizer_bayes = NonconvexOptimizer(
        n_dims=len(B),
        bounds=bounds,
        solver='BAYESIAN',
        solver_params={'n_calls': 50}
    )
    
    # Optimize RSP
    start_time = time.time()
    w_opt, val_opt = optimizer_bayes.maximize(lambda w: sp.RSP(w)[1])
    solve_time = time.time() - start_time
    
    print(f"Optimal w: {np.round(w_opt, 4)}")
    print(f"RSP value: {val_opt:.6f}")
    print(f"Time: {solve_time:.4f} seconds")

    x_rsp_proj = sp.SP(w_opt)[0]
    print(f"x_rsp_proj: {np.array(x_rsp_proj, dtype=int)}")

    # brute force optimization
    n_samples = 10000
    op = MPAssortOriginal(u, r, B, distr, C, samples=distr.random_sample((n_samples, len(u)+1)))
    bf_optimizer = BruteForceOptimizer(N=N, C=C, num_cores=8)
    x_op, _ = bf_optimizer.maximize(op)
    print(f"x_op: {np.array(x_op, dtype=int)}")
    
    op_val_for_x_op = op(x_op)
    op_val_for_x_rsp = op(x_rsp_proj)
    
    print(f"Original Pi value for x_op: {op_val_for_x_op:.4f}")
    print(f"Original Pi value for x_rsp_proj: {op_val_for_x_rsp:.4f}")
    print(f"Relative gap: {(op_val_for_x_op - op_val_for_x_rsp)/op_val_for_x_op:.4%}") 

    w_op_sol = sp._w_x(x_op)
    print(f"The better w: {w_op_sol}")
    print(f"The RSP value: {sp.RSP(w_op_sol)[1]}")
    # x_examine = sp.SP(w_op_sol)[0]
    # print(f"The better x: {x_op}")


    