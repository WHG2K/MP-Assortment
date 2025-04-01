N = 15
C = (8,8)
B = [1, 2, 3]
distr = "GumBel"
random_seed = 2025
n_instances = 5

B_str = '_'.join(map(str, B))
file_name = f'raw_N_{N}_C_{C[0]}_{C[1]}_B_{B_str}_distr_{distr}.pkl'

print(file_name)