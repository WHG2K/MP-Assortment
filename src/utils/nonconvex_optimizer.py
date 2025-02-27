import numpy as np
from skopt import gp_minimize
import pyswarms as ps
from typing import Callable, Tuple, Optional, Dict, Any

class NonconvexOptimizer:
    """A solver class for nonconvex optimization problems with box constraints"""
    
    def __init__(self, 
                 n_dims: int,
                 bounds: Optional[np.ndarray] = None,
                 solver: str = 'PSO',
                 solver_params: Optional[Dict[str, Any]] = None):
        """Initialize the optimizer
        
        Args:
            n_dims: Number of dimensions
            bounds: Array of bounds, shape (n_dims, 2). Each row is [lower, upper]
                   If None, default to [0,1] for each dimension
            solver: Solver to use ('BAYESIAN' or 'PSO')
            solver_params: Additional parameters for the solver
        """
        self.n_dims = n_dims
        self.bounds = bounds if bounds is not None else np.array([[0, 1]] * n_dims)
        
        # Valid solvers
        self.VALID_SOLVERS = ['BAYESIAN', 'PSO']
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        self.solver = solver
        
        # Default parameters for each solver
        self.default_params = {
            'BAYESIAN': {
                'n_calls': 100,
                'n_random_starts': 10,
                'noise': 1e-10,
                'random_state': 42
            },
            'PSO': {
                'n_particles': 50,
                'max_iter': 100,
                'c1': 2.0,
                'c2': 2.0,
                'w': 0.7,
                'k': 3,
                'p': 2,
                'random_state': 42
            }
        }
        
        # Update with user-provided parameters
        self.solver_params = self.default_params[solver].copy()
        if solver_params:
            self.solver_params.update(solver_params)
    
    def maximize(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Maximize the objective function subject to box constraints"""
        if self.solver == 'BAYESIAN':
            return self._solve_bayesian(objective_fn)
        else:  # PSO
            return self._solve_PSO(objective_fn)
    
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
    
    def _solve_PSO(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Solve using Particle Swarm Optimization"""
        # Define objective function for PSO (PySwarms minimizes by default)
        def objective_function(x):
            return [-objective_fn(w) for w in x]
        
        bounds_arr = np.array(self.bounds)
        
        # Setup bounds for PySwarms
        lb = bounds_arr[:, 0]  # lower bounds
        ub = bounds_arr[:, 1]  # upper bounds
        bounds = (lb, ub)
        
        # Initialize swarm
        options = {
            'c1': self.solver_params['c1'],
            'c2': self.solver_params['c2'],
            'w': self.solver_params['w'],
            'k': self.solver_params['k'],
            'p': self.solver_params['p']
        }
        
        # Create optimizer
        optimizer = ps.single.LocalBestPSO(
            n_particles=self.solver_params['n_particles'],
            dimensions=self.n_dims,
            options=options,
            bounds=bounds,
            init_pos=None,
            velocity_clamp=None,
            vh_strategy='unmodified',
            center=1.00,
            ftol=-np.inf,
            ftol_iter=1
        )
        
        # Perform optimization
        cost, pos = optimizer.optimize(
            objective_function,
            iters=self.solver_params['max_iter'],
            verbose=False
        )
        
        # Return the best position and its objective value (negated back)
        return pos, -cost 