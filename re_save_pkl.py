import pickle
import numpy as np

if __name__ == "__main__":

    file_path = 'results/0317_sBB/N_60_C_25_25_B_1_to_10_sBB/N_60_C_25_25_B_1_to_10_sBB.pkl'

    # with open("your_file.pkl", "wb") as f:
    #     pickle.dump(data, f, protocol=0)  # 使用 ASCII 格式

    data = np.load(file_path, allow_pickle=True)
    print(data)
