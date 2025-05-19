import argparse
import numpy as np
from src.utils.distributions import GumBel
import pickle
from tqdm import tqdm
import time
from src.algorithms.models import MPAssortSurrogate, MPAssortOriginal
from src.ptas.PTAS import AO_Instance, MP_MNL_PTAS
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
    parser.add_argument('--tol_sp', type=float, default=0.0001, help='Tolerance for exact SP solution')
    parser.add_argument('--tol_rsp', type=float, default=0.0001, help='Tolerance for exact RSP solution')
    parser.add_argument('--tol_sp_avg', type=float, default=0.0001, help='Tolerance for averaged SP solution')
    parser.add_argument('--tol_rsp_avg', type=float, default=0.0001, help='Tolerance for averaged RSP solution')
    parser.add_argument('--tol_ptas', type=float, default=0.5, help='Epsilon for PTAS')
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    parser.add_argument("--randomness", type=str, default='rand', help='randomness of basket sizes')
    parser.add_argument("--file_name", type=str, default="", help='name of the file to save')

    # which algorithm to run
    parser.add_argument("--generate", action="store_true", help="Generate instances")
    parser.add_argument("--sp", action="store_true", help="Use exact SP")
    parser.add_argument("--rsp", action="store_true", help="Use exact RSP")
    parser.add_argument("--sp_avg", action="store_true", help="Use averaged SP")
    parser.add_argument("--rsp_avg", action="store_true", help="Use averaged RSP")
    parser.add_argument("--ptas", action="store_true", help="Use PTAS")
    parser.add_argument("--bf", action="store_true", help="Use brute force search")
    parser.add_argument("--gr", action="store_true", help="Use greedy search")
    parser.add_argument("--mnl", action="store_true", help="Use MNL search")
    parser.add_argument("--eval", action="store_true", help="Evaluate the performance of the algorithms")
    parser.add_argument("--all", action="store_true", help="Run all algorithms")


    return parser.parse_args()



def generate_instances(N=15, C=(8,8), B=[1,2,3], randomness='rand', distr="GumBel", random_seed=2025, n_instances=100):
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
        u = np.random.normal(0, 1, N).reshape(-1).tolist()
        w = np.exp(u)
        w_max = np.max(w)
        r = (w_max - w).reshape(-1).tolist()

        # Generate basket size distribution
        basket_sizes = B
        probs = np.random.uniform(0, 1, len(basket_sizes))
        if randomness == 'rand':
            pass
        elif randomness == 'dec':
            probs = np.sort(probs)[::-1]
        else:
            raise ValueError(f"Unsupported randomness: {randomness}")
        probs = probs / probs.sum()
        probs = probs.reshape(-1).tolist()
        B = dict(zip(basket_sizes, probs))

        # Store instance data
        instance = {
            'instance_id': i,
            'N': N,
            'C': C,
            'randomness': randomness,
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


def solve_sp(input_file, output_file, tolerance=0.05):
    """Solve SP

    Args:
        XXXXXX
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving SP for {len(instances)} instances with tolerance {tolerance}...")
    x_name = 'x_sp_' + str(tolerance)
    time_name = 'time_sp_' + str(tolerance)
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

        try:
            start_time = time.time()
            sp_obj = SP_obj(model)
            sp_ub = SP_ub(model)
            sp_lb = SP_lb(model)
            w_sp, _ = spatial_branch_and_bound_maximize(
                sp_obj, sp_lb, sp_ub,
                (box_low, box_high),
                tolerance=tolerance
            )
            inst[time_name] = time.time() - start_time
            x_sp = model.SP(w_sp, solver='gurobi')[0]
            inst[x_name] = np.array(x_sp).reshape(-1).tolist()

        except Exception as e:
            inst[time_name] = None
            inst[x_name] = None
            print(f"Error in SP for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")



def solve_rsp(input_file, output_file, tolerance=0.05):
    """Solve RSP

    Args:
        XXXXXX
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving RSP for {len(instances)} instances with tolerance {tolerance}...")
    x_name = 'x_rsp_' + str(tolerance)
    time_name = 'time_rsp_' + str(tolerance)
    for idx, inst in enumerate(tqdm(instances)):
        if inst['distr'] == "GumBel":
            distr = GumBel()
        else:
            raise ValueError(f"Unsupported distribution: {inst['distr']}")

        model = MPAssortSurrogate(u=inst['u'], r=inst['r'], B=inst['B'], 
                                distr=distr, C=inst['C'])
        w_range = np.array(model._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)

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
            inst[time_name] = time.time() - start_time
            x_rsp = model.SP(w_rsp, solver='gurobi')[0]
            inst[x_name] = np.array(x_rsp).reshape(-1).tolist()


        except Exception as e:
            inst[time_name] = None
            inst[x_name] = None
            print(f"Error in RSP for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")






def solve_avg_sp(input_file, output_file, tolerance=0.05):
    """Solve averaged SP

    Args:
        XXXXXX
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving the averaged SP for {len(instances)} instances with tolerance {tolerance}...")
    x_name = 'x_avg_sp_' + str(tolerance)
    time_name = 'time_avg_sp_' + str(tolerance)
    for idx, inst in enumerate(tqdm(instances)):

        # initialize model and get box constraints
        if inst['distr'] == "GumBel":
            distr = GumBel()
        else:
            raise ValueError(f"Unsupported distribution: {inst['distr']}")

        # change B to a point mass
        B = inst['B']
        avg_B = sum(x * p for x, p in B.items())
        model = MPAssortSurrogate(u=inst['u'], r=inst['r'], B=avg_B, 
                                distr=distr, C=inst['C'])
        w_range = np.array(model._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)


        try:
            start_time = time.time()
            sp_obj = SP_obj(model)
            sp_ub = SP_ub(model)
            sp_lb = SP_lb(model)
            w_sp, _ = spatial_branch_and_bound_maximize(
                sp_obj, sp_lb, sp_ub,
                (box_low, box_high),
                tolerance=tolerance
            )
            inst[time_name] = time.time() - start_time
            x_avg_sp = model.SP(w_sp, solver='gurobi')[0]
            inst[x_name] = np.array(x_avg_sp).reshape(-1).tolist()

        except Exception as e:
            inst[time_name] = None
            inst[x_name] = None
            print(f"Error in averaged SP for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")





def solve_avg_rsp(input_file, output_file, tolerance=0.05):
    """Solve averaged RSP

    Args:
        XXXXXX
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving the averaged RSP for {len(instances)} instances with tolerance {tolerance}...")
    x_name = 'x_avg_rsp_' + str(tolerance)
    time_name = 'time_avg_rsp_' + str(tolerance)
    for idx, inst in enumerate(tqdm(instances)):
        if inst['distr'] == "GumBel":
            distr = GumBel()
        else:
            raise ValueError(f"Unsupported distribution: {inst['distr']}")

        # change B to a point mass
        B = inst['B']
        avg_B = sum(x * p for x, p in B.items())
        model = MPAssortSurrogate(u=inst['u'], r=inst['r'], B=avg_B, 
                                distr=distr, C=inst['C'])
        w_range = np.array(model._get_box_constraints())
        box_low = np.array(w_range[:, 0]).reshape(-1)
        box_high = np.array(w_range[:, 1]).reshape(-1)

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
            inst[time_name] = time.time() - start_time
            x_avg_rsp = model.SP(w_rsp, solver='gurobi')[0]
            inst[x_name] = np.array(x_avg_rsp).reshape(-1).tolist()

        except Exception as e:
            inst[time_name] = None
            inst[x_name] = None
            print(f"Error in averaged RSP for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")



def solve_ptas(input_file, output_file, tolerance=0.5):

    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving PTAS for {len(instances)} instances with tolerance {tolerance}...")
    x_name = 'x_ptas_' + str(tolerance)
    time_name = 'time_ptas_' + str(tolerance)
    for idx, inst in enumerate(tqdm(instances)):
        try:
            if inst['distr'] == "GumBel":
                distr = GumBel()
            else:
                raise ValueError(f"Unsupported distribution: {inst['distr']}")

            # Create ptas solver
            m = max(inst['B'].keys())
            lambda_ = [0.0] * (m + 1)  # index 0 unused, just set to 0.0
            for k in range(1, m + 1):
                lambda_[k] = inst['B'].get(k, 0.0)
            weights = np.exp(inst['u']).reshape(-1).tolist()
            ao_instance = AO_Instance(inst['N'], m, lambda_, weights, inst['r'], inst['C'][0])

            # Solve with PTAS
            ptas_solver = MP_MNL_PTAS(ao_instance)
            start_time = time.time()
            _ = ptas_solver.solve(tolerance)
            S_x = ptas_solver.best_S
            x = np.zeros(inst['N'], dtype=int)
            x[S_x] = 1
            x_ptas = x.tolist()
            inst[time_name] = time.time() - start_time
            inst[x_name] = x_ptas

        except Exception as e:
            inst[time_name] = None
            inst[x_name] = None
            print(f"Error in PTAS for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")





def solve_brute_force(input_file, output_file, num_workers=1):
    """Solve brute force
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving brute force for {len(instances)} instances...")
    for idx, inst in enumerate(tqdm(instances)):
        try:
            if inst['distr'] == "GumBel":
                distr = GumBel()
            else:
                raise ValueError(f"Unsupported distribution: {inst['distr']}")

            # Create original problem solver
            op = MPAssortOriginal(inst['u'], inst['r'], inst['B'], distr, inst['C'])
            op_obj = OP_obj(op)


            # Solve with brute force
            bf_optimizer = BruteForceOptimizer(N=inst['N'], C=inst['C'], num_cores=num_workers)
            start_time = time.time()
            x_op, _ = bf_optimizer.maximize(op_obj)
            inst['time_opt'] = time.time() - start_time
            inst['x_opt'] = np.array(x_op).reshape(-1).tolist()

        except Exception as e:
            inst['time_opt'] = None
            inst['x_opt'] = None
            print(f"Error in brute force for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")



def solve_greedy(input_file, output_file):
    """Solve the greedy heuristic
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving the greedy solution for {len(instances)} instances...")
    for idx, inst in enumerate(tqdm(instances)):
        try:
            if inst['distr'] == "GumBel":
                distr = GumBel()
            else:
                raise ValueError(f"Unsupported distribution: {inst['distr']}")

            # Create original problem solver
            op = MPAssortOriginal(inst['u'], inst['r'], inst['B'], distr, inst['C'])
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
            print(f"Error in greedy for instance {inst['instance_id']}: {e}")

    # Save updated instances
    with open(output_file, 'wb') as f:
        pickle.dump(instances, f)
    print(f"\nUpdated instances saved to {output_file}\n")
        


def solve_mnl(input_file, output_file):
    """Solve MNL
    """
    # Load instances
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    print(f"Solving MNL for {len(instances)} instances...")
    for idx, inst in enumerate(tqdm(instances)):
        if inst['distr'] == "GumBel":
            distr = GumBel()
        else:
            raise ValueError(f"Unsupported distribution: {inst['distr']}")

        try:
            # Create MNL problem
            mnl = MNL(inst['u'], inst['r'])

            # Solve MNL problem
            start_time = time.time()
            x_mnl, _ = mnl.solve(inst['C'])
            inst['time_mnl'] = time.time() - start_time
            inst['x_mnl'] = np.array(x_mnl).reshape(-1).tolist()

        except Exception as e:
            inst['time_mnl'] = None
            inst['x_mnl'] = None
            print(f"Error in MNL for instance {inst['instance_id']}: {e}")

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

    if args.all:
        args.generate = True
        args.sp = True
        args.rsp = True
        args.sp_avg = True
        args.rsp_avg = True
        args.ptas = True
        args.bf = True
        args.gr = True
        args.mnl = True
        args.eval = True


    # parameters
    N = args.N
    C = tuple(args.C)
    B = args.B
    n_instances = args.n_instances
    distr = "GumBel"
    randomness = args.randomness
    random_seed = 2025

    B_str = '_'.join(map(str, B))
    if args.file_name:
        file_name = args.file_name
    else:
        file_name = f'raw_{randomness}_N_{N}_C_{C[0]}_{C[1]}_B_{B_str}_distr_{distr}.pkl'

    # Generate and save instances
    if args.generate:
        instances = generate_instances(
            N=N,
            C=C,
            B=B,
            randomness=randomness,
            distr=distr,
            random_seed=random_seed,
            n_instances=n_instances
        )
        save_instances(instances, file_name=file_name)

    # solve mnl
    if args.mnl:
        solve_mnl(input_file=file_name, output_file=file_name)
    # Solve sp via spatial branch-and-bound
    if args.sp:
        solve_sp(input_file=file_name, output_file=file_name, tolerance=args.tol_sp)
    # Solve rsp via spatial branch-and-bound
    if args.rsp:
        solve_rsp(input_file=file_name, output_file=file_name, tolerance=args.tol_rsp)
    # Solve the averaged sp via spatial branch-and-bound
    if args.sp_avg:
        solve_avg_sp(input_file=file_name, output_file=file_name, tolerance=args.tol_sp_avg)
    # Solve the averaged rsp via spatial branch-and-bound
    if args.rsp_avg:
        solve_avg_rsp(input_file=file_name, output_file=file_name, tolerance=args.tol_rsp_avg)
    # Solve the ptas
    if args.ptas:
        solve_ptas(input_file=file_name, output_file=file_name, tolerance=args.tol_ptas)
    # solve greedy
    if args.gr:
        solve_greedy(input_file=file_name, output_file=file_name)
    # solve brute force
    if args.bf:
        solve_brute_force(input_file=file_name, output_file=file_name)
    # evaluate revenue
    if args.eval:
        evaluate_revenue(input_file=file_name, output_file=file_name)