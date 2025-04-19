import pickle
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Load pickle file
    # with open(r'results\0402_runtime\raw_N_100_C_12_12_B_1_distr_GumBel_tol_0.01.pkl', 'rb') as f:
    #     instances = pickle.load(f)
    # file_name = r'results\0410_accuracy\bai_et_al_setting\tol_0.001\N60_no_bf\raw_rand_N_60_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form'
    # file_name = r'test\raw_rand_N_30_C_12_12_B_1_2_3_distr_GumBel_tol_0.0001_close_form_new'
    # file_name = r'test\raw_dec_N_15_C_8_8_B_1_2_distr_GumBel_tol_0.0001_close_form_new'
    # file_name = r'bai_et_al_setting/tol_0.0001/N15_bf/raw_rand_N_15_C_12_12_B_1_2_3_distr_GumBel_tol_0.001_close_form'
    # file_name = r'results\0410_accuracy\scale_B\tol_0.0001\raw_rand_N_15_C_8_12_B_3_distr_GumBel_tol_0.0001_close_form'
    # file_name = r'test\raw_rand_N_15_C_8_12_B_3_distr_GumBel_tol_0.0001_close_form'
    file_name = r'test\raw_rand_N_60_C_12_20_B_3_distr_GumBel_tol_0.0001_close_form_new'
    with open(file_name + '.pkl', 'rb') as f:
        instances = pickle.load(f)
    
    # instances = instances.to_dict(orient="records")
    # print(instances[0])

    # for key in instances[0].keys():
    #     print(key, type(instances[0][key]))

    # # Convert to DataFrame and select columns
    # df = pd.DataFrame(instances)[['instance_id', 'pi_x_exact_sp', 'pi_x_exact_rsp', 'pi_x_clustered_sp', 'pi_x_clustered_rsp', 'pi_x_mnl', 'pi_x_gr',
    #                               'time_exact_sp', 'time_exact_rsp', 'time_clustered_sp', 'time_clustered_rsp']]

    # # Save to Excel
    # df.to_excel('check_sBB_003.xlsx', index=False)

    df = pd.DataFrame(instances)
    df.to_excel(file_name + '.xlsx', index=False)
