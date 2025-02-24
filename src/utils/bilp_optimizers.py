import numpy as np
import os
from typing import Tuple, Optional
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BinaryProgramSolver:
    """A solver class for binary (0-1) integer linear programs"""
    
    def __init__(self, solver: str = 'pulp'):
        """Initialize the solver
        
        Args:
            solver: Solver to use ('pulp' or 'gurobi')
        """
        self.solver = solver.lower()
        if self.solver not in ['pulp', 'gurobi']:
            raise ValueError("Solver must be either 'pulp' or 'gurobi'")
        
        if self.solver == 'gurobi':
            try:
                import gurobipy as gp
                from gurobipy import GRB
                self.gp = gp
                self.GRB = GRB
                self.gurobi_license = os.getenv('GUROBI_LICENSE_PATH')
                if self.gurobi_license is None:
                    raise ValueError("GUROBI_LICENSE_PATH not found in environment variables")
            except ImportError:
                raise ImportError("Gurobi is not available. Please install gurobipy.")
    
    def maximize(self, 
                c: np.ndarray,
                A: Optional[np.ndarray] = None,
                b: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, str]:
        """Solve a binary maximization problem
        
        Args:
            c: Objective coefficients vector
            A: Matrix of inequality constraints (Ax <= b)
            b: RHS vector of inequality constraints
        
        Returns:
            Tuple[float, np.ndarray, str]: (
                optimal value,
                optimal solution vector,
                solution status
            )
        """
        if self.solver == 'pulp':
            return self._solve_pulp(c, A, b)
        else:  # gurobi
            return self._solve_gurobi(c, A, b)
    
    def _solve_pulp(self, c: np.ndarray, A: Optional[np.ndarray], b: Optional[np.ndarray]) -> Tuple[float, np.ndarray, str]:
        """Solve using PuLP"""
        N = len(c)
        model = LpProblem(name="binary_program", sense=LpMaximize)
        x = [LpVariable(name=f'x_{i}', cat=LpBinary) for i in range(N)]
        
        model += lpSum(c[i] * x[i] for i in range(N))
        
        if A is not None and b is not None:
            for i in range(len(b)):
                model += lpSum(A[i,j] * x[j] for j in range(N)) <= b[i]
        
        solver = PULP_CBC_CMD(msg=False)
        model.solve(solver)
        
        status = LpStatus[model.status]
        if status == 'Optimal':
            sol = np.array([x[i].value() for i in range(N)])
            obj_val = model.objective.value()
            return obj_val, sol, status
        return 0.0, np.zeros(N), status
    
    def _solve_gurobi(self, c: np.ndarray, A: Optional[np.ndarray], b: Optional[np.ndarray]) -> Tuple[float, np.ndarray, str]:
        """Solve using Gurobi"""
        N = len(c)
        os.environ['GRB_LICENSE_FILE'] = self.gurobi_license
        
        with self.gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            # env.setParam('Threads', 16)
            env.start()
            
            with self.gp.Model(env=env) as model:
                x = model.addVars(N, vtype=self.GRB.BINARY, name="x")
                
                model.setObjective(
                    self.gp.quicksum(c[i] * x[i] for i in range(N)),
                    self.GRB.MAXIMIZE
                )
                
                if A is not None and b is not None:
                    for i in range(len(b)):
                        model.addConstr(
                            self.gp.quicksum(A[i,j] * x[j] for j in range(N)) <= b[i]
                        )
                
                model.optimize()
                
                # Map Gurobi status codes to meaningful strings
                status_map = {
                    self.GRB.OPTIMAL: 'Optimal',
                    self.GRB.INFEASIBLE: 'Infeasible',
                    self.GRB.INF_OR_UNBD: 'Infeasible or Unbounded',
                    self.GRB.UNBOUNDED: 'Unbounded',
                    self.GRB.TIME_LIMIT: 'Time Limit Reached',
                    self.GRB.NODE_LIMIT: 'Node Limit Reached',
                    self.GRB.SOLUTION_LIMIT: 'Solution Limit Reached',
                    self.GRB.INTERRUPTED: 'Interrupted',
                    self.GRB.NUMERIC: 'Numeric Issues'
                }
                
                status = status_map.get(model.status, f'Other Status: {model.status}')
                
                if model.status == self.GRB.OPTIMAL:
                    sol = np.array([x[i].X for i in range(N)])
                    obj_val = model.objVal
                else:
                    sol = np.zeros(N)
                    obj_val = 0.0
                
                return obj_val, sol, status