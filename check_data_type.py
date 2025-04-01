import pandas as pd
import pickle

def check_first_row_types(df):
    """Check the data types of each entry in the first row of the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        dict: A dictionary with column names as keys and their respective data types as values.
    """
    first_row = df.iloc[0]
    return {column: type(first_row[column]) for column in df.columns}


with open('data_sBB_003.pkl', 'rb') as f:
    instances = pickle.load(f)
    df = pd.DataFrame(instances)

print(check_first_row_types(df))
