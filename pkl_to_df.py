import pickle
import pandas as pd

if __name__ == "__main__":

    # Load pickle file
    with open('data.pkl', 'rb') as f:
        instances = pickle.load(f)

    # Convert to DataFrame and select columns
    df = pd.DataFrame(instances)[['instance_id', 'pi_x_op', 'pi_x_sp_exact', 'pi_x_sp_clustered', 'pi_x_rsp_exact', 'pi_x_rsp_clustered', 'pi_x_mnl', 'pi_x_gr']]

    # Save to Excel
    df.to_excel('check.xlsx', index=False) 