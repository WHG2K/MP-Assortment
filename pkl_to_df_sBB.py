import pickle
import pandas as pd

if __name__ == "__main__":

    # Load pickle file
    with open('raw_N_20_C_8_8_B_1_2_3_distr_GumBel.pkl', 'rb') as f:
        instances = pickle.load(f)

    # # Convert to DataFrame and select columns
    # df = pd.DataFrame(instances)[['instance_id', 'pi_x_exact_sp', 'pi_x_exact_rsp', 'pi_x_clustered_sp', 'pi_x_clustered_rsp', 'pi_x_mnl', 'pi_x_gr',
    #                               'time_exact_sp', 'time_exact_rsp', 'time_clustered_sp', 'time_clustered_rsp']]

    # # Save to Excel
    # df.to_excel('check_sBB_003.xlsx', index=False)

    df = pd.DataFrame(instances)
    print(df)
