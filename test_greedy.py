import numpy as np
from src.utils.greedy import GreedyOptimizer
from src.algorithms.models import MNL

if __name__ == "__main__":
    # Set random seed
    # np.random.seed(42)
    
    # Problem parameters
    N = 10
    C = (6, 6)
    
    # Generate random instance
    u = np.random.normal(0, 1, N)  # utilities
    w = np.exp(u)
    w_max = np.max(w)
    r = w_max - w  # revenues
    
    # Create MNL instance
    mnl = MNL(u, r)
    
    # Create optimizer instance
    GR = GreedyOptimizer(N=N, C=C)
    
    # Run maximize
    x_gr, val_gr = GR.maximize(mnl)
    

    print(f"\nGreedy Solution:")
    print(f"x = {x_gr}")
    print(f"obj = {val_gr:.6f}")
    print(f"num selected = {sum(x_gr)}")

    # optimal solution
    x_opt, val_opt = mnl.solve(C)
    print(f"\nOptimal Solution:")
    print(f"x = {x_opt}")
    print(f"obj = {val_opt:.6f}")
    print(f"num selected = {sum(x_opt)}")

