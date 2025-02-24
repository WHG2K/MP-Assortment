import numpy as np
import itertools
import multiprocessing
from typing import Tuple, Callable

class BruteForceOptimizer:
    """A brute force optimizer for finding optimal binary vector that maximizes a given function"""
    
    def __init__(self, N: int, C: Tuple[int, int], num_cores: int = 1):
        """Initialize the optimizer
        
        Args:
            N: Length of binary vector
            C: Tuple of (min_cardinality, max_cardinality)
            num_cores: Number of CPU cores to use for parallel processing
        """
        self.N = N
        self.C = C
        self.num_cores = min(num_cores, multiprocessing.cpu_count())
    
    def _binary_to_int(self, binary_vec: np.ndarray) -> int:
        """Convert binary vector to integer for memory efficiency"""
        return int(''.join(map(str, binary_vec)), 2)
    
    def _int_to_binary(self, num: int, N: int) -> np.ndarray:
        """Convert integer back to binary vector"""
        return np.array([int(bit) for bit in bin(num)[2:].zfill(N)])
    
    def _chunk_combinations(self) -> list:
        """Generate and chunk all valid combinations"""
        all_combos = []
        for cardinal in range(self.C[0], self.C[1] + 1):
            all_combos.extend(list(itertools.combinations(range(self.N), cardinal)))
            
        # Convert combinations to integers for memory efficiency
        all_combos_int = []
        for combo in all_combos:
            binary_vec = np.zeros(self.N, dtype=np.int64)
            binary_vec[list(combo)] = 1
            all_combos_int.append(self._binary_to_int(binary_vec))
            
        # Divide combinations into chunks for parallel processing
        divide_by = min(self.num_cores, len(all_combos_int))
        chunk_size = len(all_combos_int) // divide_by
        remainder = len(all_combos_int) % divide_by
        
        chunks = []
        start = 0
        for i in range(divide_by):
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append(all_combos_int[start:start + size])
            start += size
            
        return chunks
    
    def _evaluate_chunk(self, chunk: list, objective_fn: Callable, result_queue: multiprocessing.Queue):
        """Evaluate a chunk of combinations"""
        max_value = float('-inf')
        max_solution = None
        
        for combo_int in chunk:
            x = self._int_to_binary(combo_int, self.N)
            value = objective_fn(x)
            
            if value > max_value:
                max_value = value
                max_solution = x
        
        result_queue.put((max_solution, max_value))
    
    def maximize(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        """Find binary vector that maximizes the objective function
        
        Args:
            objective_fn: Function that takes binary vector and returns float value
            
        Returns:
            Tuple[np.ndarray, float]: (optimal binary vector, optimal value)
        """
        chunks = self._chunk_combinations()
        result_queue = multiprocessing.Queue()
        processes = []
        
        # Start parallel processes
        for chunk in chunks:
            process = multiprocessing.Process(
                target=self._evaluate_chunk,
                args=(chunk, objective_fn, result_queue)
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        # Collect results
        max_value = float('-inf')
        optimal_solution = None
        
        while not result_queue.empty():
            solution, value = result_queue.get()
            if value > max_value:
                max_value = value
                optimal_solution = solution
        
        return optimal_solution, max_value 