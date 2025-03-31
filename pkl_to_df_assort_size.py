import pickle
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Load pickle file
    with open('N_15_C_4_12_B_1_2_3_(linux_read).pkl', 'rb') as f:
        instances = pickle.load(f)

    # Convert to DataFrame and select columns
    df = pd.DataFrame(instances)[['x_sp_exact', 'x_sp_clustered', 'x_mnl', 'x_gr']]
    df['size_sp_exact'] = df['x_sp_exact'].apply(lambda x: np.sum(np.array(x).reshape(-1)))
    df['size_sp_clustered'] = df['x_sp_clustered'].apply(lambda x: np.sum(np.array(x).reshape(-1)))
    df['size_mnl'] = df['x_mnl'].apply(lambda x: np.sum(np.array(x).reshape(-1)))
    df['size_gr'] = df['x_gr'].apply(lambda x: np.sum(np.array(x).reshape(-1)))

    # Save to Excel
    df.to_excel('check_assort_size.xlsx', index=False) 