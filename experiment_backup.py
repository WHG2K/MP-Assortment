import numpy as np
from scipy.special import softmax
from src.utils.distributions import GumBel
import pickle
from tqdm import tqdm
import time
from src.algorithms.models import MPAssortSurrogate, MPAssortOriginal
from src.algorithms.BB import branch_and_bound
from src.utils.brute_force import BruteForceOptimizer
from src.algorithms.BnB_functions_utils import RSP_obj, RSP_ub, RSP_lb, SP_obj, SP_ub, SP_lb, OP_obj
from dotenv import load_dotenv
import os
load_dotenv(override=True)


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

def solve_instances(input_file, output_file, num_workers=8):
    """Solve instances using SP BnB and RSP BnB methods
    
    Args:
        pkl_file: Name of pickle file containing instances
    """
    # Load instances
    with open(input_file, 'rb') as f:
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
                    tolerance=0.03,
                    min_box_size=0.01,
                    num_workers=num_workers
                )
                inst['time_sp_bnb'] = time.time() - start_time
                inst['sp_bnb'] = model.SP(w_sp, solver='gurobi')[0]
                success_count_sp += 1

            except Exception as e:
                inst['time_sp_bnb'] = None
                inst['sp_bnb'] = None
                print(f"Error in SP BnB: {e}")
        else:
            print(f"SP BnB solution already exists for instance {inst['instance_id']}")
            success_count_sp += 1

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
                    tolerance=0.03,
                    min_box_size=0.01,
                    num_workers=num_workers
                )
                inst['time_rsp_bnb'] = time.time() - start_time
                inst['rsp_bnb'] = model.SP(w_rsp, solver='gurobi')[0]
                success_count_rsp += 1
            
            except Exception as e:
                inst['time_rsp_bnb'] = None
                inst['rsp_bnb'] = None
                inst['error'] = str(e)

        else:
            print(f"RSP BnB solution already exists for instance {inst['instance_id']}")
            success_count_rsp += 1
        
    success_rate_sp = success_count_sp / len(instances)
    success_rate_rsp = success_count_rsp / len(instances)
    print(f"Success rate SP: {success_rate_sp:.2%}, Success rate RSP: {success_rate_rsp:.2%}")
    
    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}")

def solve_brute_force(input_file='instances.pkl', output_file='instances.pkl', n_samples=10000, num_workers=8):
    """Solve instances using brute force and calculate optimality gaps
    
    Args:
        pkl_file: Name of pickle file containing instances
        n_samples: Number of samples for MPAssortOriginal
        num_cores: Number of cores for parallel computation
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)
    
    print(f"Solving {len(instances)} instances with brute force...")
    for idx, inst in enumerate(tqdm(instances)):
        try:
            # Create solver instances
            if inst['distr'] == "GumBel":
                distr = GumBel()
            else:
                raise ValueError(f"Unsupported distribution: {inst['distr']}")
                
            # Create original problem solver
            op = MPAssortOriginal(inst['u'], inst['r'], inst['B'], distr, inst['C'])

            op_obj = OP_obj(op, distr.random_sample((n_samples, inst['N']+1)))
            
            # Solve with brute force
            bf_optimizer = BruteForceOptimizer(N=inst['N'], C=inst['C'], num_cores=num_workers)
            
            start_time = time.time()
            x_op, val_op = bf_optimizer.maximize(op_obj)
            inst['time_bf'] = time.time() - start_time
            inst['x_bf'] = x_op
            
            # Calculate optimality gaps if SP and RSP solutions exist
            if 'sp_bnb' in inst and inst['sp_bnb'] is not None:
                inst['gap_sp'] = (val_op - op_obj(inst['sp_bnb'])) / val_op
            else:
                inst['gap_sp'] = None
                
            if 'rsp_bnb' in inst and inst['rsp_bnb'] is not None:
                inst['gap_rsp'] = (val_op - op_obj(inst['rsp_bnb'])) / val_op
            else:
                inst['gap_rsp'] = None
            
        except Exception as e:
            inst['time_bf'] = None
            inst['x_bf'] = None
            inst['gap_sp'] = None
            inst['gap_rsp'] = None
            print(f"Error in brute force: {e}")
    
    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}")

if __name__ == "__main__":

    # check gurobi home and license
    gurobi_home = os.getenv("GUROBI_HOME")
    license_file = os.getenv("GRB_LICENSE_FILE")
    print(f"Gurobi home: {gurobi_home}")
    print(f"License path: {license_file}")

    # Generate and save instances
    if True:
        instances = generate_instances(N=15, C=(9,9), distr="GumBel", random_seed=2025, n_instances=2)
        save_instances(instances, file_name='instances_0.pkl')
    
    # Solve instances
    if False:
        solve_instances(input_file='instances_0.pkl', output_file='instances_1.pkl', num_workers=4)
    
    # Solve with brute force and calculate gaps
    if True:
        solve_brute_force(input_file='instances_0.pkl', output_file='instances_try.pkl', n_samples=1000, num_workers=8)