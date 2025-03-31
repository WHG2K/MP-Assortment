import numpy as np
import gurobipy as gp
from gurobipy import GRB

# def wasserstein_1(x: np.ndarray, y: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
#     """Compute the Wasserstein-1 distance between two discrete distributions.
    
#     Args:
#         x: Support points of the first distribution (shape: n)
#         y: Support points of the second distribution (shape: m)
#         p: Probabilities of the first distribution (shape: n)
#         q: Probabilities of the second distribution (shape: m)
        
#     Returns:
#         float: The Wasserstein-1 distance
        
#     Note:
#         Solves the linear programming problem:
#         min sum_{i,j} pi_{ij} |x_i - y_j|
#         s.t. sum_j pi_{ij} = p_i for all i
#              sum_i pi_{ij} = q_j for all j
#              pi_{ij} >= 0
#     """
#     n = len(x)
#     m = len(y)
    
#     # Create cost matrix
#     C = np.abs(x.reshape(-1, 1) - y.reshape(1, -1))
    
#     # Create LP problem
#     prob = LpProblem("Wasserstein_Distance", LpMinimize)
    
#     # Create variables
#     pi = LpVariable.dicts("pi", 
#                          ((i, j) for i in range(n) for j in range(m)), 
#                          lowBound=0)
    
#     # Objective function
#     prob += lpSum(C[i,j] * pi[i,j] for i in range(n) for j in range(m))
    
#     # Constraints for first distribution
#     for i in range(n):
#         prob += lpSum(pi[i,j] for j in range(m)) == p[i]
    
#     # Constraints for second distribution
#     for j in range(m):
#         prob += lpSum(pi[i,j] for i in range(n)) == q[j]
    
#     # Solve the problem
#     prob.solve(PULP_CBC_CMD(msg=0))
    
#     # Return optimal value
#     return value(prob.objective)


def wasserstein_1_barycenter_1D(x: np.ndarray, p: np.ndarray) -> float:
    """Compute the Wasserstein-1 barycenter of a set of distributions.
    
    Args:
        x: Support points of the distributions (shape: n)
        p: Probabilities of the distributions (shape: n)
        
    """
    m = len(x)
    if (len(p) != m):
        raise ValueError("x and p must have the same length")

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        
        with gp.Model(env=env) as model:
            a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="a")
            t = model.addVars(m, lb=0, name="t")

            # objective: Minimize sum(p_i * t_i)
            model.setObjective(gp.quicksum(p[i] * t[i] for i in range(m)), GRB.MINIMIZE)

            # constraints: t_i >= |x_i - a|
            for i in range(m):
                model.addConstr(t[i] >= x[i] - a, name=f"abs1_{i}")
                model.addConstr(t[i] >= a - x[i], name=f"abs2_{i}")

            # optimize
            model.optimize()
            
            # Map Gurobi status codes to meaningful strings
            status_map = {
                GRB.OPTIMAL: 'Optimal',
                GRB.INFEASIBLE: 'Infeasible',
                GRB.INF_OR_UNBD: 'Infeasible or Unbounded',
                GRB.UNBOUNDED: 'Unbounded',
                GRB.TIME_LIMIT: 'Time Limit Reached',
                GRB.NODE_LIMIT: 'Node Limit Reached',
                GRB.SOLUTION_LIMIT: 'Solution Limit Reached',
                GRB.INTERRUPTED: 'Interrupted',
                GRB.NUMERIC: 'Numeric Issues'
            }
            
            status = status_map.get(model.status, f'Other Status: {model.status}')
            
            if model.status == GRB.OPTIMAL:
                return a.X
            else:
                raise ValueError(f"Optimization failed with status: {status}")

if __name__ == "__main__":
    # test_wasserstein() 
    x = np.array([1, 2, 3])
    p = np.array([0.1, 0.2, 0.7])
    print(wasserstein_1_barycenter_1D(x, p))
