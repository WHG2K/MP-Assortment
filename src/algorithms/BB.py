import numpy as np
from collections import deque
from multiprocessing import Pool, cpu_count
    


# Branch and Bound Algorithm
def process_node(args):
    """Process a single node in parallel
    
    Args:
        args: tuple containing (box_low, box_high, f_lb, f_ub, f, min_box_size, lb)
        
    Returns:
        tuple: (new_boxes, lb_current, ub_current, x)
        where new_boxes is a list of (box_low, box_high) pairs for new nodes
    """
    box_low, box_high, f_lb, f_ub, min_box_size, lb = args
    
    # Calculate bounds
    lb_current, x = f_lb(box_low, box_high)
    ub_current = f_ub(box_low, box_high)
    
    # Check if node should be pruned
    box_size = np.max(np.abs(box_high - box_low))
    if ub_current < lb or box_size < min_box_size:
        return [], lb_current, ub_current, x
        
    # Split the node
    lengths = [box_high[i] - box_low[i] for i in range(len(box_low))]
    max_dim = np.argmax(lengths)
    
    midpoint = (box_low[max_dim] + box_high[max_dim]) / 2
    
    left_box_low = np.copy(box_low)
    left_box_high = np.copy(box_high)
    left_box_high[max_dim] = midpoint

    right_box_low = np.copy(box_low)
    right_box_high = np.copy(box_high)
    right_box_low[max_dim] = midpoint
    
    new_boxes = [(left_box_low, left_box_high), 
                 (right_box_low, right_box_high)]
    
    return new_boxes, lb_current, ub_current, x

def branch_and_bound(f, f_lb, f_ub, box_low, box_high, tolerance=0.05, min_box_size=1e-1, num_workers=1):
    """Branch and Bound algorithm with parallel processing
    
    Args:
        f: objective function
        f_lb: lower bound function
        f_ub: upper bound function
        box_low: lower bounds of initial box
        box_high: upper bounds of initial box
        tolerance: convergence tolerance
        min_box_size: minimum box size for splitting
        num_workers: number of parallel workers (default: number of CPU cores - 1)
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Initialize optimal solution and best lower bound
    best_solution = None
    best_objective = -np.inf
    lb = -np.inf   # Global lower bound
    
    queue = deque([(box_low, box_high)])
    layer = 0
    
    # Create process pool
    with Pool(num_workers) as pool:
        while queue:
            layer_len = len(queue)
            ub = -np.inf
            
            # Prepare arguments for parallel processing
            process_args = [(box_low, box_high, f_lb, f_ub, min_box_size, lb) 
                          for box_low, box_high in [queue.popleft() for _ in range(layer_len)]]
            
            # Process nodes in parallel
            results = pool.map(process_node, process_args)
            
            # Process results and update queue
            new_queue = deque()
            for new_boxes, lb_current, ub_current, x in results:
                # Update bounds
                ub = max(ub, ub_current)
                if lb_current > lb:
                    lb = lb_current
                    best_solution = x
                    best_objective = f(x)
                
                # Add new boxes if not pruned
                new_queue.extend(new_boxes)
            
            queue = new_queue
            print(f"Layer={layer:2d}, ub={ub:.4f}, lb={lb:.4f}")
            layer += 1
            
            if ub - lb < tolerance * ub:
                print(f"Converged at layer={layer:2d}, ub={ub:.4f}, lb={lb:.4f}, tol={(tolerance * ub):.4f}")
                return best_solution, f(best_solution)
    
    print(f"Reaching min box size")
    return best_solution, f(best_solution)
