import argparse
import numpy as np
# from scipy.special import softmax
from src.utils.distributions import GumBel
import pickle
from tqdm import tqdm
import time
from src.algorithms.models import MPAssortSurrogate, MPAssortOriginal
from src.algorithms.sBB import spatial_branch_and_bound_maximize
from src.utils.brute_force import BruteForceOptimizer
from src.utils.greedy import GreedyOptimizer
from src.algorithms.models import MNL
from src.algorithms.sBB_functions_utils import RSP_obj, RSP_ub, RSP_lb, SP_obj, SP_ub, SP_lb, OP_obj
from dotenv import load_dotenv
import os



def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    # Adding arguments
    parser.add_argument('--N', type=int, default=15, help='Value of N')
    parser.add_argument('--C', type=int, nargs=2, default=(12, 12), help='Tuple of C (two integers)')
    parser.add_argument('--B', type=int, nargs='+', default=[1, 2, 3], help='List of B values')
    parser.add_argument('--n_instances', type=int, default=100, help='Number of instances to generate')
    parser.add_argument('--random_seed', type=int, default=2025, help='Random seed for instance generation')
    parser.add_argument('--distr', type=str, default="GumBel", help='Distribution type')
    parser.add_argument('--tol_exact', type=float, default=0.01, help='Tolerance for exact SP/RSP solution')
    parser.add_argument('--node', type=str, default=None, help='which node to run on')

    return parser.parse_args()



def generate_instances(N=15, C=(8,8), B=[1,2,3], distr="GumBel", random_seed=2025, n_instances=100):
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
        u = np.random.normal(0, 1, N).flatten().tolist()
        w = np.exp(u)
        w_max = np.max(w)
        r = (w_max - w).flatten().tolist()

        # Generate basket size distribution
        basket_sizes = B
        probs = np.random.uniform(0, 1, len(basket_sizes))
        probs = probs / probs.sum()
        probs = probs.flatten().tolist()
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

def save_instances(instances, file_name):
    """Save instances to pickle file

    Args:
        instances: List of instance dictionaries
        file_name: Name of pickle file to save
    """
    with open(file_name, 'wb') as f:
        pickle.dump(instances, f)



def solve_exact_rsp_fractional(input_file, output_file, tolerance=0.05):
    """Solve the exact RSP via sBB

    Args:
        XXXXXX
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving the exact RSP for {len(instances)} instances via spatial branch-and-bound...")
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

        if ("x_bar_exact_rsp" not in inst) or (inst["x_bar_exact_rsp"] is None):
            try:
                start_time = time.time()
                rsp_obj = RSP_obj(model)
                rsp_ub = RSP_ub(model)
                rsp_lb = RSP_lb(model)
                w_rsp, _ = spatial_branch_and_bound_maximize(
                    rsp_obj, rsp_lb, rsp_ub,
                    (box_low, box_high),
                    tolerance=tolerance
                )
                inst['time_exact_rsp'] = time.time() - start_time
                x_bar_exact_rsp = model.RSP(w_rsp, solver='gurobi')[0]
                # print(x_bar_exact_rsp, type(x_bar_exact_rsp[0]))
                inst['x_bar_exact_rsp'] = np.array(x_bar_exact_rsp).flatten().tolist()
                success_count_rsp += 1

            except Exception as e:
                inst['time_exact_rsp'] = None
                inst['x_bar_exact_rsp'] = None
                print(f"Error in solving the exact RSP: {e}")


        else:
            print(f"The exact RSP fractional solution already exists for instance {inst['instance_id']}")
            success_count_rsp += 1

    # print success rate
    print(f"Success: {success_count_rsp}/{len(instances)}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")



if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # choose environment
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')

    # check gurobi home and license
    gurobi_home = os.getenv("GUROBI_HOME")
    license_file = os.getenv("GRB_LICENSE_FILE")
    print(f"Gurobi home: {gurobi_home}")
    print(f"License path: {license_file}")

    # parameters
    N = args.N
    C = tuple(args.C)
    B = args.B
    n_instances = args.n_instances
    distr = "GumBel"
    random_seed = 2025

    B_str = '_'.join(map(str, B))
    file_name = f'raw_LP_relax_N_{N}_C_{C[0]}_{C[1]}_B_{B_str}_distr_{distr}_tol_{args.tol_exact}.pkl'

    # Generate and save instances
    if True:
        instances = generate_instances(
            N=N,
            C=C,
            B=B,
            distr=distr,
            random_seed=random_seed,
            n_instances=n_instances
        )
        save_instances(instances, file_name=file_name)

    # Solve the exact rsp via spatial branch-and-bound
    if True:
        solve_exact_rsp_fractional(input_file=file_name, output_file=file_name, tolerance=args.tol_exact)