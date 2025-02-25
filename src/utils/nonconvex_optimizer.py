import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize
from typing import Callable, Tuple, Optional, Dict, Any

class NonconvexOptimizer:
    """A solver class for nonconvex optimization problems with box constraints"""
    
    def __init__(self, 
                 n_dims: int,
                 bounds: Optional[np.ndarray] = None,
                 solver: str = 'BFGS',
                 solver_params: Optional[Dict[str, Any]] = None):
        """Initialize the optimizer
        
        Args:
            n_dims: Number of dimensions
            bounds: Array of bounds, shape (n_dims, 2). Each row is [lower, upper]
                   If None, default to [0,1] for each dimension
            solver: Solver to use ('BFGS' or 'BAYESIAN')
            solver_params: Additional parameters for the solver
        """
        self.n_dims = n_dims
        self.bounds = bounds if bounds is not None else np.array([[0, 1]] * n_dims)
        
        # Validate solver
        valid_solvers = ['BFGS', 'BAYESIAN']
        if solver not in valid_solvers:
            raise ValueError(f"Solver must be one of {valid_solvers}")
        self.solver = solver
        
        # Set default parameters
        self.default_params = {
            'BFGS': {
                'method': 'L-BFGS-B',
                'options': {'maxiter': 1000}
            },
            'BAYESIAN': {
                'n_calls': 50,
                'n_random_starts': 10,
                'noise': 1e-10,
                'random_state': 42
            }
        }
        
        # Update with user-provided parameters
        self.solver_params = self.default_params[solver].copy()
        if solver_params:
            self.solver_params.update(solver_params)
    
    def maximize(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Maximize the objective function subject to box constraints"""
        if self.solver == 'BFGS':
            return self._solve_BFGS(objective_fn)
        elif self.solver == 'BAYESIAN':
            return self._solve_bayesian(objective_fn)
        else:
            raise ValueError(f"Solver {self.solver} not implemented")
    
    def _solve_BFGS(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Solve using L-BFGS-B method"""
        def neg_objective(x):
            return -objective_fn(x)
        
        bounds = [(b[0], b[1]) for b in self.bounds]
        x0 = np.mean(self.bounds, axis=1)
        
        result = minimize(
            neg_objective,
            x0,
            bounds=bounds,
            **self.solver_params
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        return result.x, -result.fun
    
    def _solve_bayesian(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Solve using Bayesian optimization"""
        def neg_objective(x):
            return -objective_fn(np.array(x))
        
        # Convert bounds to list of tuples for skopt
        bounds = [(b[0], b[1]) for b in self.bounds]
        
        result = gp_minimize(
            neg_objective,
            bounds,
            **self.solver_params
        )
        
        return np.array(result.x), -result.fun 