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
    parser.add_argument('--folder_path', type=str, default=None, help='folder path to save instances')

    return parser.parse_args()



def solve_greedy(input_file, output_file):
    """Solve instances using greedy heuristic
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving {len(instances)} instances with greedy heuristic...")
    for idx, inst in enumerate(tqdm(instances)):
        try:
            # Create solver instances
            if inst['distr'] == "GumBel":
                distr = GumBel()
            else:
                raise ValueError(f"Unsupported distribution: {inst['distr']}")

            # Create original problem solver
            op = MPAssortOriginal(inst['u'], inst['r'], inst['B'], distr, inst['C'])
            # random_comps = distr.random_sample((n_samples, inst['N']+1))
            op_obj = OP_obj(op)

            # Solve with greedy
            greedy_optimizer = GreedyOptimizer(N=inst['N'], C=inst['C'])
            start_time = time.time()
            x_gr, _ = greedy_optimizer.maximize(op_obj)
            inst['time_gr'] = time.time() - start_time
            inst['x_gr'] = np.array(x_gr).reshape(-1).tolist()

        except Exception as e:
            inst['time_gr'] = None
            inst['x_gr'] = None
            # inst['pi_x_op'] = None
            print(f"Error in brute force: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")


def evaluate_revenue(input_file, output_file):
    """Evaluate the revenue of the instances
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    # get list of solutions
    columns = instances[0].keys()
    x_names = [col for col in columns if col.startswith('x_')]

    # evaluate revenue
    print(f"Evaluating revenue for {len(instances)} instances...")
    for idx, inst in enumerate(tqdm(instances)):

        # Create solver instances
        if inst['distr'] == "GumBel":
            distr = GumBel()
        else:
            raise ValueError(f"Unsupported distribution: {inst['distr']}")
        
        # create original problem solver
        op = MPAssortOriginal(inst['u'], inst['r'], inst['B'], distr, inst['C'])
        op_obj = OP_obj(op)

        # evaluate revenue
        for x in x_names:
            if inst[x] is not None:
                inst[f'pi_{x}'] = float(op_obj(inst[x]))
    
    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")
        


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    folder_path = args.folder_path
    
    # check gurobi home and license
    gurobi_home = os.getenv("GUROBI_HOME")
    license_file = os.getenv("GRB_LICENSE_FILE")
    print(f"Gurobi home: {gurobi_home}")
    print(f"License path: {license_file}")

    # folder_path = r"results\0410_accuracy\correct\test"
    for filename in os.listdir(folder_path):
        print(filename)  
        if filename.startswith("raw") and filename.endswith(".pkl"):  
            file_path = os.path.join(folder_path, filename)

        solve_greedy(input_file=file_path, output_file=file_path)
        evaluate_revenue(input_file=file_path, output_file=file_path)
