import pickle
import numpy as np
import pandas as pd

def load_instances(file_name='instances.pkl'):
    """Load instances from pickle file
    
    Args:
        file_name: Name of pickle file to load
        
    Returns:
        pd.DataFrame: DataFrame containing instances
    """
    with open(file_name, 'rb') as f:
        df = pickle.load(f)
    return df

if __name__ == "__main__":
    # Load instances
    instances = load_instances("data.pkl")

    # df = pd.DataFrame(instances)
    # print(df)
    for row in instances:
        # print(row)
        for key, value in row.items():
            print(key, value)
