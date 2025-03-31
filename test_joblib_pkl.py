import joblib
import pickle
import os
import numpy as np


if __name__ == "__main__":

    file_path = 'results/0317_sBB/N_60_C_25_25_B_1_to_10_sBB/N_60_C_25_25_B_1_to_10_sBB.pkl'

    with open(file_path, 'rb') as f:
        content = f.read(1000)  # 读取前 100 个字节
        print(content)
        data = pickle.load(f)

    # print(data)

