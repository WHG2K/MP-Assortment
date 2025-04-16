import numpy as np
from typing import Union, Tuple, List, Dict
from src.utils.distributions import Distribution
from src.utils.bilp_optimizers import BinaryProgramSolver
from src.utils.lp_optimizers import LinearProgramSolver
from itertools import combinations
# from scipy.integrate import quad_vec
import time


class MPAssortSurrogate:
    """Multi-Purchase Assortment Optimization Surrogate Algorithm
    
    This is an implementation of the surrogate problem (SP) algorithm for multi-purchase assortment 
    optimization problem, based on the paper "Multi-Purchase Assortment Optimization 
    Under a General Random Utility Model" (Abdallah et al., 2024).
    
    Attributes:
        u (np.ndarray): Intrinsic values of products
        r (np.ndarray): Unit revenues of products
        B (Dict[int, float]): Basket size distribution (maps size to probability)
        distr (Distribution): Distribution of random utility error term
        C (Tuple[int, int]): Cardinality constraint on assortment size
        N (int): Number of products
    """
    
    def __init__(self, 
                 u: Union[List[float], np.ndarray],
                 r: Union[List[float], np.ndarray],
                 B: Union[int, Dict[int, float]],
                 distr: Distribution,
                 C: Union[int, Tuple[int, int]]):
        """Initialize the algorithm class
        
        Args:
            u: List or array of product intrinsic values
            r: List or array of product unit revenues
            B: Either a single basket size or a distribution over basket sizes
               If single value: must be positive integer
               If distribution: must be dict mapping non-negative integers to probabilities
            distr: Distribution of random utility error term
            C: Cardinality constraint, can be an integer or (min, max) tuple
        
        Raises:
            AssertionError: When lengths of u and r are not equal
            ValueError: When parameter types or values are invalid
        """
        # Validate input parameters
        if not isinstance(u, (list, np.ndarray)) or not isinstance(r, (list, np.ndarray)):
            raise ValueError("u and r must be lists or numpy arrays")
        
        # Convert to numpy arrays for unified processing
        self.u = np.array(u, dtype=float)
        self.r = np.array(r, dtype=float)
        
        # Validate lengths
        assert len(self.u) == len(self.r), "lengths of u and r must be equal"
        
        # Validate and store B parameter
        if isinstance(B, dict):
            self.B = B
        else:
            self.B = {B: 1.0}

        # if isinstance(B, int):
        #     if B <= 0:
        #         raise ValueError("B as single value must be a positive integer")
        #     self.B = {B: 1.0}  # Convert to distribution with probability 1
        # elif isinstance(B, dict):
        #     # Validate the distribution
        #     if not B:
        #         raise ValueError("B as distribution cannot be empty")
        #     if not all(isinstance(k, int) and k >= 0 for k in B.keys()):
        #         raise ValueError("B distribution keys must be non-negative integers")
        #     if not all(isinstance(p, (int, float)) and p >= 0 for p in B.values()):
        #         raise ValueError("B distribution probabilities must be non-negative")
        #     total_prob = sum(B.values())
        #     if not np.isclose(total_prob, 1.0):
        #         raise ValueError(f"B distribution probabilities must sum to 1, got {total_prob}")
        #     self.B = B
        # else:
        #     raise ValueError("B must be either an integer or a dictionary distribution")
        
        if not isinstance(distr, Distribution):
            raise ValueError("distr must be an instance of Distribution class")
        self.distr = distr
        
        # Handle cardinality constraint
        if not isinstance(C, (int, tuple)):
            raise ValueError("C must be an integer or tuple")
        if isinstance(C, int):
            self.C = (C, C)
        else:
            if len(C) != 2 or C[0] > C[1] or C[0] < 0:
                raise ValueError("C as tuple must be in (min, max) format with 0 ≤ min ≤ max")
            self.C = C
            
        # Store sorted basket sizes and corresponding probabilities
        self._Bs = np.array(sorted(self.B.keys()))
        self._B_probs = np.array([self.B[k] for k in self._Bs])
        
        # Store number of products
        self.N = len(self.u) 


    def SP(self, w: Union[List[float], np.ndarray, float], solver: str = 'pulp') -> Tuple[np.ndarray, float]:
        """ Compute SP(w)
        
        Args:
            w: vector of length |B|.
            solver: Solver to use for the binary program
        
        Returns:
            Tuple[np.ndarray, float]: (
                x: best assortment under SP(w),
                sp_w: Value of the SP(w)
            )
            
        Raises:
            ValueError: If length of w doesn't match the basket size distribution support
        """
        # Convert input to 1D numpy array
        if np.isscalar(w):
            w = np.array([w])
        else:
            w = np.asarray(w).flatten()
        
        # Validate input
        if len(w) != len(self.B):
            raise ValueError(f"Length of w ({len(w)}) must match the number of support points in B ({len(self.B)})")
        
        # Compute LP parameters
        c, A, b = self._compute_LP_parameters(w)
        
        # Create solver and solve the binary program
        bilp_solver = BinaryProgramSolver(solver=solver)
        sp_w, x, status = bilp_solver.maximize(c, A, b)
        
        if status != 'Optimal':
            raise ValueError(f"Failed to solve SP: {status}")
        
        # Convert solution to integer array
        x = np.round(x).astype(int)
        
        return x, sp_w
    

    def RSP(self, w: Union[List[float], np.ndarray, float], solver: str = 'pulp') -> Tuple[np.ndarray, float]:
        """ Compute RSP(w)
        
        Args:
            w: vector of length |B|.
            solver: Solver to use for the binary program
        
        Returns:
            Tuple[np.ndarray, float]: (
                x: best assortment under RSP(w),
                rsp_w: Value of the RSP(w)
            )
            
        Raises:
            ValueError: If length of w doesn't match the basket size distribution support
        """
        # Convert input to 1D numpy array
        if np.isscalar(w):
            w = np.array([w])
        else:
            w = np.asarray(w).flatten()
        
        # Validate input
        if len(w) != len(self.B):
            raise ValueError(f"Length of w ({len(w)}) must match the number of support points in B ({len(self.B)})")
        
        # Compute LP parameters
        c, A, b = self._compute_LP_parameters(w)
        
        # Create solver and solve the binary program
        lp_solver = LinearProgramSolver(solver=solver)
        rsp_w, x, status = lp_solver.maximize(c, A, b)
        
        if status != 'Optimal':
            raise ValueError(f"Failed to solve SP: {status}")
        
        # Convert solution to integer array
        # x = np.round(x).astype(int)
        
        return x, rsp_w

    def _probs_U_exceed_w(self, w: np.ndarray) -> np.ndarray:
        """Compute P(u[j] + X > w[i]) for all pairs of i,j
        
        Args:
            w: Weight vector (1D numpy array)
            
        Returns:
            np.ndarray: Matrix of shape (len(w), len(u)) containing probabilities
        """
        return 1 - self.distr.cdf(w[:, None] - self.u[None, :])  # Shape: (|B|, N)
    
    def _probs_buying_surrogate(self, w: np.ndarray) -> np.ndarray:
        """ Compute sum_i P(B=b_i) P(u[j] + X > max(w[i], Y))
        
        Args:
            w: Weight vector (1D numpy array)
            
        Returns:
            np.ndarray: vector of length N.
        """

        return self.distr._compute_c_vector(self.u, w, self._B_probs)
        # # Create integrand object
        # integrand = _Integrand_purchasing_probs(self.u, w, self._B_probs, self.distr)
        
        # # Compute integral from -10 to 10
        # # The interval [-10, 10] is chosen as a reasonable approximation of (-∞, ∞)
        # result, _ = quad_vec(
        #         integrand,
        #         a=-10, b=10,
        #         epsabs=1e-3,  # absolute error tolerance
        #         epsrel=1e-3,  # relative error tolerance
        #         full_output=False,
        #         workers=1  # set 1 just for now.
        #     )
        
        # return np.array(result).reshape(-1)

    def _compute_LP_parameters(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the parameters (c, A, b) for the binary integer linear program
        
        Args:
            w: Weight vector (1D numpy array)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (
                c: objective coefficients (shape: N),
                A: constraint matrix (shape: (|B|+2) x N),
                b: RHS of constraints (shape: |B|+2)
            )
        """
        
        # Compute objective coefficients c
        c = self._probs_buying_surrogate(w) * self.r
        
        # Compute P(u[j] + X > w[i]) for constraints
        constraint_probs = self._probs_U_exceed_w(w)
        
        # Construct constraint matrix A
        A = np.vstack([
            constraint_probs,  # First |B| rows are P(u[j] + X > w[i])
            np.ones(self.N),   # Cardinality upper bound
            -np.ones(self.N)   # Cardinality lower bound
        ])
        
        # Compute RHS vector b
        b = np.concatenate([
            self._Bs,
            [self.C[1], -self.C[0]]
        ])
        
        return c, A, b
    
    def _w_x(self, x: np.ndarray) -> np.ndarray:
        """Compute w=w(x)
        
        Args:
            x: assortment vector of length N.
        
        Returns:
            np.ndarray: vector of length |B|.
        """
        x = np.array(x).reshape(-1)
        w_low = np.array([-5.0] * len(self._Bs))
        w_high = np.array([5.0] * len(self._Bs))
        i = 0
        while (i < 15):
            w = (w_low + w_high) / 2
            probs_matrix = self._probs_U_exceed_w(w)
            viol_cons = probs_matrix @ x - self._Bs
            w_high[viol_cons < 0] = w[viol_cons < 0]
            w_low[viol_cons > 0] = w[viol_cons > 0]
            i += 1

        return w_high
    
    def _pi_hat(self, x: np.ndarray) -> float:
        """ Compute pi_hat(x)
        
        Args:
            x: vector of length |B|.
        
        Returns:
            float: pi_hat(x).
        """
        # find w=w(x)
        w = self._w_x(x)
        # compute pi_hat(x)
        return np.dot(x, self._probs_buying_surrogate(w) * self.r)
    
    def _get_box_constraints(self) -> List[Tuple[float, float]]:
        """Get box constraints for w variables"""
        # get the rankings of utilities
        sorted_indices = np.argsort(-self.u)

        # get the C[0] lowest and C[1] highest utilities
        x_min = np.zeros(len(self.u))
        x_min[sorted_indices[-self.C[0]:]] = 1
        x_max = np.zeros(len(self.u))
        x_max[sorted_indices[:self.C[1]]] = 1

        # get the box constraints for w
        w_low = self._w_x(x_min)
        w_high = self._w_x(x_max)
        
        return [(w_low[i], w_high[i]) for i in range(len(self._Bs))]
    

    def __call__(self, x: np.ndarray) -> float:
        """Compute the value of pi(x). Mainly served for parallel computation."""
        return self._pi_hat(x)
    



class _Integrand_purchasing_probs:
    """A class to encapsulate the integrand function for purchasing probability computation, 
    mainly used to enable parallel computation.
    
    This class implements the integrand function needed to compute P(u[j] + X > max(w[i], Y))
    where X and Y are independent random variables following the same distribution.
    
    Attributes:
        u (np.ndarray): Utility values vector
        w (np.ndarray): Weight values vector
        distr (Distribution): Distribution object that provides pdf and cdf methods
    """
    
    def __init__(self, u, w, B_probs, distr):
        """Initialize the integrand with utility values, weights, and distribution
        
        Args:
            u (np.ndarray): Vector of utility values
            w (np.ndarray): Vector of weight values
            distr (Distribution): Distribution object with pdf and cdf methods
        """
        self.u = u
        self.w = w
        self.distr = distr
        self._B_probs = B_probs

    def __call__(self, x):
        """Compute the integrand for parallel computation
        """
        u = self.u
        w = self.w
        distr = self.distr
        B_probs = self._B_probs

        # compute u + x
        u_plus_x = u + x
        
        # indicator matrix for w[i] < u[j] + x condition
        indicator = np.dot(B_probs, w[:, None] < u_plus_x)
        
        # Get CDF values for u + x
        cdf_values = distr.cdf(u_plus_x)

        return (cdf_values * indicator)  * distr.pdf(x)











class MPAssortOriginal:
    """Multi-Purchase Assortment Optimization Original Problem (OP)
    
    This class implements the original problem (OP) for multi-purchase assortment 
    optimization, based on the paper "Multi-Purchase Assortment Optimization 
    Under a General Random Utility Model" (Abdallah et al., 2024).
    
    Attributes:
        u (np.ndarray): Intrinsic values of products
        r (np.ndarray): Unit revenues of products
        B (Dict[int, float]): Basket size distribution (maps size to probability)
        distr (Distribution): Distribution of random utility error term
        C (Tuple[int, int]): Cardinality constraint on assortment size
        N (int): Number of products
    """
    
    def __init__(self, 
                 u: Union[List[float], np.ndarray],
                 r: Union[List[float], np.ndarray],
                 B: Union[int, Dict[int, float]],
                 distr: Distribution,
                 C: Union[int, Tuple[int, int]]):
        """Initialize the algorithm class
        
        Args:
            u: List or array of product intrinsic values
            r: List or array of product unit revenues
            B: Either a single basket size or a distribution over basket sizes
               If single value: must be positive integer
               If distribution: must be dict mapping non-negative integers to probabilities
            distr: Distribution of random utility error term
            C: Cardinality constraint, can be an integer or (min, max) tuple
        """
        # Validate input parameters
        if not isinstance(u, (list, np.ndarray)) or not isinstance(r, (list, np.ndarray)):
            raise ValueError("u and r must be lists or numpy arrays")
        
        # Convert to numpy arrays for unified processing
        self.u = np.array(u, dtype=float)
        self.r = np.array(r, dtype=float)
        
        # Validate lengths
        assert len(self.u) == len(self.r), "lengths of u and r must be equal"
        
        # Validate and store B parameter
        if isinstance(B, int):
            if B <= 0:
                raise ValueError("B as single value must be a positive integer")
            self.B = {B: 1.0}  # Convert to distribution with probability 1
        elif isinstance(B, dict):
            # Validate the distribution
            if not B:
                raise ValueError("B as distribution cannot be empty")
            if not all(isinstance(k, int) and k >= 0 for k in B.keys()):
                raise ValueError("B distribution keys must be non-negative integers")
            if not all(isinstance(p, (int, float)) and p >= 0 for p in B.values()):
                raise ValueError("B distribution probabilities must be non-negative")
            total_prob = sum(B.values())
            if not np.isclose(total_prob, 1.0):
                raise ValueError(f"B distribution probabilities must sum to 1, got {total_prob}")
            self.B = B
        else:
            raise ValueError("B must be either an integer or a dictionary distribution")
        
        if not isinstance(distr, Distribution):
            raise ValueError("distr must be an instance of Distribution class")
        self.distr = distr
        
        # Handle cardinality constraint
        if not isinstance(C, (int, tuple)):
            raise ValueError("C must be an integer or tuple")
        if isinstance(C, int):
            self.C = (C, C)
        else:
            if len(C) != 2 or C[0] > C[1] or C[0] < 0:
                raise ValueError("C as tuple must be in (min, max) format with 0 ≤ min ≤ max")
            self.C = C
            
        # Store sorted basket sizes and corresponding probabilities
        self._Bs = np.array(sorted(self.B.keys()))
        self._B_probs = np.array([self.B[k] for k in self._Bs])
        
        # Store number of products
        self.N = len(self.u)

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate samples from the random utility error term"""
        return self.distr.random_sample((n_samples, self.N+1))
    
    def _pi_monte_carlo(self, x: np.ndarray, random_comps: np.ndarray) -> float:
        """Compute the value of pi(x) using Monte Carlo simulation"""
        # preprocess the random components
        RU = random_comps[:, :self.N] + self.u
        RU -= random_comps[:, -1:]

        x = np.array(x).reshape(-1)
        
        # Keep only columns where x[i]==1
        RU_x = RU[:, x > 0.5]
        selected_r = self.r[x > 0.5]
        
        # Create mask for positive utilities
        positive_mask = RU_x > 0
        
        # For each basket size b, create selection mask for top b items
        selections = {}
        purchase_probs = np.zeros(len(selected_r))
        
        for b in self._Bs:
            # Get rank of each element (in descending order)
            ranks = np.argsort(np.argsort(-RU_x, axis=1), axis=1)
            # Combine rank and positivity conditions
            selections[b] = (ranks < b) & positive_mask
            # Count selections for each product
            product_selections = np.sum(selections[b], axis=0)
            purchase_probs += product_selections * self.B[b]
        
        n_samples = len(RU_x)
        purchase_probs /= n_samples
        
        # Compute weighted sum of selections
        weighted_sum = np.zeros_like(RU_x, dtype=float)
        for b, prob in zip(self._Bs, self._B_probs):
            weighted_sum += selections[b] * prob
        
        # Compute expected revenue
        pi_x_monte_carlo = np.sum(weighted_sum * selected_r[None, :]) / n_samples
        
        return pi_x_monte_carlo


    def Get_Choice_Prob_MP_MNL(self, S, i):
        # get the parameters from model
        B = self.B
        m = max(B.keys())
        lambda_ = [0.0] * (m + 1)  # index 0 unused, just set to 0.0
        for k in range(1, m + 1):
            lambda_[k] = B.get(k, 0.0)
        weights = np.exp(self.u).reshape(-1).tolist()
        
        w_i = weights[i]
        D_choice_prob = {}
        size_S = len(S)
        for j in range(1,min(size_S,m)+1):
            D_choice_prob[j] = {}
            for k in range(j,min(size_S,m)+1):
                left_overs = [c for c in list(combinations(S,size_S-(min(size_S,m)-k))) if i in c]
                if j==1:
                    for assort in left_overs:
                        total_weight = 1+sum(weights[prod] for prod in assort)
                        D_choice_prob[j][assort] = w_i/total_weight
                else:
                    for assort in left_overs:
                        total_weight = 1+sum(weights[prod] for prod in assort)
                        next_m = 0
                        for prod in assort:
                            nc_prob = weights[prod]/total_weight
                            if prod!=i:
                                new_assort = tuple([nc for nc in assort if nc!=prod])
                                next_m+=(nc_prob*D_choice_prob[j-1][new_assort])
                                
                        D_choice_prob[j][assort] = w_i/total_weight + next_m
        
                        
        choice_prob = D_choice_prob[min(size_S,m)][tuple(S)]*sum(lambda_[min(size_S,m):])
        for j in range(1,min(size_S,m)):
            choice_prob+=(lambda_[j]*D_choice_prob[j][tuple(S)])
                
        return choice_prob


    def _pi(self, x):
        # get the parameters from model
        B = self.B
        m = max(B.keys())
        lambda_ = [0.0] * (m + 1)  # index 0 unused, just set to 0.0
        for k in range(1, m + 1):
            lambda_[k] = B.get(k, 0.0)
        weights = np.exp(self.u).reshape(-1).tolist()
        S = [i for i, val in enumerate(x) if val > 0.99]
        prices = self.r
        
        total_rev =0
        for i in S:
            choice_prob = self.Get_Choice_Prob_MP_MNL(S,i)
            total_rev+=(prices[i]*choice_prob)
        return total_rev





class MNL:
    """Multinomial Logit Model for assortment optimization.
    
    Attributes:
        u: Utility vector
        v: Revenue vector
        N: Number of products
    """
    def __init__(self, u: np.ndarray, r: np.ndarray):
        """Initialize MNL model.
        
        Args:
            u: Utility vector
            v: Revenue vector
            
        Raises:
            ValueError: If utility and revenue vectors have different lengths
        """
        u = np.array(u).reshape(-1)
        r = np.array(r).reshape(-1)
        if len(u) != len(r):
            raise ValueError(f"Utility vector (length {len(u)}) and revenue vector (length {len(r)}) must have the same length")
            
        self.u = u
        self.r = r
        self.N = len(u)
        
    def solve(self, C: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        """Solve the MNL assortment optimization problem with cardinality constraints."""
        # Import Gurobi here instead of storing it as instance attribute
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as e:
            print(f"Error: {e}")
            return None, None
            
        # Handle cardinality constraint
        if isinstance(C, int):
            if C < 0 or C > self.N:
                raise ValueError(f"C must be between 0 and {self.N}")
            C = (C, C)  # Convert to tuple format
        elif isinstance(C, tuple):
            if len(C) != 2 or C[0] > C[1] or C[0] < 0 or C[1] > self.N:
                raise ValueError(f"C as tuple must be in (min, max) format with 0 ≤ min ≤ max ≤ {self.N}")
        else:
            raise ValueError("C must be an integer or tuple")
            
        # construct linear program to solve it
        N = self.N
        r = self.r
        v = np.exp(self.u)

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            
            with gp.Model(env=env) as model:
                w = model.addVars(N+1, lb=0, name="w")
                
                # set objective
                model.setObjective(
                    gp.quicksum(r[i] * w[i] for i in range(N)),
                    GRB.MAXIMIZE
                )

                # probability constraints
                model.addConstr(
                    gp.quicksum(w[i] for i in range(N+1)) <= 1
                )
                # cardinality constraints
                model.addConstr(
                    gp.quicksum(w[i]/v[i] for i in range(N)) >= C[0] * w[N]
                )
                model.addConstr(
                    gp.quicksum(w[i]/v[i] for i in range(N)) <= C[1] * w[N]
                )
                # additional constraints
                for i in range(N):
                    model.addConstr(
                        w[i]/v[i] <= w[N]
                    )

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
                    W = np.array([w[i].X for i in range(N)])
                    x = np.where(W > 1e-8, 1, 0)
                    obj_val = model.objVal
                else:
                    raise ValueError(f"Optimization failed with status: {status}")
                
                return x, obj_val

    def __call__(self, x: np.ndarray) -> float:
        """Calculate the expected revenue of the MNL model.
        
        Args:
            x: Assortment vector
            
        Returns:
            float: Expected revenue
        """
        x = np.array(x).reshape(-1)
        v = np.exp(self.u)
        return np.sum(self.r * v * x) / (1 + np.sum(v * x))

