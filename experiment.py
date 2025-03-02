import numpy as np
from scipy.special import softmax
from src.utils.distributions import GumBel
import pickle
from tqdm import tqdm
import time
from src.algorithms.models import MPAssortSurrogate
from src.algorithms.BB import branch_and_bound
from src.utils.lp_optimizers import LinearProgramSolver
from src.algorithms.BnB_functions_utils import RSP_obj, RSP_ub, RSP_lb, SP_obj, SP_ub, SP_lb


def generate_instances(N=15, C=(8,8), distr="GumBel", random_seed=2025, n_instances=100):
    """Generate multiple problem instances
    
    Args:
        N: Number of products
        C: Tuple of (lower_bound, upper_bound) for cardinality constraint
        distr: Distribution object (default: GumBel)
        random_seed: Random seed for reproducibility
        n_instances: Number of instances to generate
        
    Returns:
        list: List of instance dictionaries
    """
    # Set global random seed
    np.random.seed(random_seed)
    
    instances = []
    print("Generating problem instances...")
    for i in tqdm(range(n_instances)):
        # Generate utilities and revenues
        u = np.random.normal(0, 1, N)
        w = np.exp(u)
        w_max = np.max(w)
        r = w_max - w
        
        # Generate basket size distribution
        basket_sizes = [1, 2, 3]
        probs = np.random.normal(0, 1, len(basket_sizes))
        probs = softmax(probs)
        B = dict(zip(basket_sizes, probs))
        
        # Store instance data
        instance = {
            'instance_id': i,
            'N': N,
            'C': C,
            'distr': distr,
            'u': u,
            'r': r,
            'B': B
        }
        instances.append(instance)
    
    return instances

def save_instances(instances, file_name='instances.pkl'):
    """Save instances to pickle file
    
    Args:
        instances: List of instance dictionaries
        file_name: Name of pickle file to save
    """
    with open(file_name, 'wb') as f:
        pickle.dump(instances, f)

def solve_instances(pkl_file='instances.pkl'):
    """Solve instances using SP BnB and RSP BnB methods
    
    Args:
        pkl_file: Name of pickle file containing instances
    """
    # Load instances
    with open(pkl_file, 'rb') as f:
        instances = pickle.load(f)
    
    print(f"Solving {len(instances)} instances...")
    success_count_sp = 0
    success_count_rsp = 0
    for idx, inst in enumerate(tqdm(instances)):
            
        # initialize model and get box constraints
        if inst['distr'] == "GumBel":
            distr = GumBel()
        else:
            raise ValueError(f"Unsupported distribution: {inst['distr']}")
        
        model = MPAssortSurrogate(u=inst['u'], r=inst['r'], B=inst['B'], 
                                distr=distr, C=inst['C'])
        w_range = np.array(model._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)
            

        #######################################
        #### Solve SP via branch-and-bound ####
        #######################################

        if ("sp_bnb" not in inst) or (inst["sp_bnb"] is None):
            try:
                start_time = time.time()
                sp_obj = SP_obj(model)
                sp_ub = SP_ub(model)
                sp_lb = SP_lb(model)
                w_sp, _ = branch_and_bound(
                    sp_obj, sp_lb, sp_ub,
                    box_low, box_high,
                    tolerance=0.05,
                    min_box_size=0.01,
                    num_workers=8
                )
                inst['time_sp_bnb'] = time.time() - start_time
                inst['sp_bnb'] = model.SP(w_sp, solver='gurobi')[0]
                success_count_sp += 1

            except Exception as e:
                inst['time_sp_bnb'] = None
                inst['sp_bnb'] = None
                print(f"Error in SP BnB: {e}")

        ########################################
        #### Solve RSP via branch-and-bound ####
        ########################################

        if ("rsp_bnb" not in inst) or (inst["rsp_bnb"] is None):
            try:
                start_time = time.time()
                rsp_obj = RSP_obj(model)
                rsp_ub = RSP_ub(model)
                rsp_lb = RSP_lb(model)
                w_rsp, _ = branch_and_bound(
                    rsp_obj, rsp_lb, rsp_ub,
                    box_low, box_high,
                    tolerance=0.05,
                    min_box_size=0.01,
                    num_workers=8
                )
                inst['time_rsp_bnb'] = time.time() - start_time
                inst['rsp_bnb'] = model.SP(w_rsp, solver='gurobi')[0]
                success_count_rsp += 1
            
            except Exception as e:
                inst['time_rsp_bnb'] = None
                inst['rsp_bnb'] = None
                inst['error'] = str(e)
        
    success_rate_sp = success_count_sp / len(instances)
    success_rate_rsp = success_count_rsp / len(instances)
    print(f"Success rate SP: {success_rate_sp:.2%}, Success rate RSP: {success_rate_rsp:.2%}")
    
    # Save updated instances
    with open(pkl_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {pkl_file}")

if __name__ == "__main__":

    # Generate and save instances
    if False:
        instances = generate_instances(N=15, C=(8,8), distr="GumBel", random_seed=2025, n_instances=3)
        save_instances(instances, file_name='instances.pkl')
    
    # Solve instances
    if True:
        solve_instances(pkl_file='instances.pkl')