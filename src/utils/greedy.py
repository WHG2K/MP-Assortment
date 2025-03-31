import numpy as np
from typing import Tuple, Union, Callable
from multiprocessing import Pool

class GreedyOptimizer:
    """Greedy algorithm for solving optimization problems with cardinality constraints.
    
    Attributes:
        N: Number of products
        C: Cardinality constraint (min, max)
        num_cores: Number of CPU cores to use for parallel computation
    """
    
    def __init__(self, 
                 N: int, 
                 C: Union[int, Tuple[int, int]], 
                 num_cores: int = 1):
        """Initialize the optimizer
        
        Args:
            N: Number of products
            C: Cardinality constraint, can be an integer or (min, max) tuple
            num_cores: Number of CPU cores to use for parallel computation
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.N = N
        
        # Handle cardinality constraint
        if isinstance(C, int):
            if C < 0 or C > N:
                raise ValueError(f"C must be between 0 and {N}")
            self.C = (C, C)
        elif isinstance(C, tuple):
            if len(C) != 2 or C[0] > C[1] or C[0] < 0 or C[1] > N:
                raise ValueError(f"C as tuple must be in (min, max) format with 0 ≤ min ≤ max ≤ {N}")
            self.C = C
        else:
            raise ValueError("C must be an integer or tuple")
            
        self.num_cores = num_cores 


    def _one_step_search_list(self, x):
        if x.sum() == self.N:
            raise ValueError("No more solutions to search")
        else:
            # generate all possible solutions by adding to x
            x = np.round(x).astype(int)
            next_searches = []
            for i in range(len(x)):
                if x[i] == 0:  # 仅考虑当前为 0 的位置
                    y = x.copy()
                    y[i] = 1
                    next_searches.append(y)
            return next_searches


    def maximize(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Maximize the objective function using a greedy algorithm
        
        Args:
            objective_fn: Objective function to maximize
            max_iter: Maximum number of iterations
        """
        # star from zero solution
        x = np.zeros(self.N, dtype=int)
        if self.C[0] > 0:
            global_val = float('-inf')
        else:
            global_val = objective_fn(x)
            # print(f"layer: {0}, global_val: {global_val}")


        while x.sum() < self.C[1]:
            next_searches = self._one_step_search_list(x)
            best_val = float('-inf')

            for y in next_searches:
                val = objective_fn(y)
                if val > best_val:
                    best_val = val
                    best_y = y
            
            # print(f"layer: {best_y.sum()}, best_val: {best_val}, global_val: {global_val}")
            
            if (best_y.sum() > self.C[0]) and (best_val < global_val):
                return x, objective_fn(x)
            else:
                global_val = best_val
                x = best_y

        return x, objective_fn(x)

                    
            
