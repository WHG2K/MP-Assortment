import pickle
import pandas as pd

if __name__ == "__main__":

    # Load pickle file
    with open('instances_1.pkl', 'rb') as f:
        instances = pickle.load(f)

    # Convert to DataFrame and select columns
    df = pd.DataFrame(instances)[['instance_id', 'gap_sp', 'gap_rsp', 'time_sp_bnb', 'time_rsp_bnb']]

    # Save to Excel
    df.to_excel('N_15_C_8_8_distr_GumBel.xlsx', index=False) 