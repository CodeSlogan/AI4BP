import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def get_filenames_in_directory(directory):
    filenames = []
    # 遍历目录中的每个项
    for item in os.listdir(directory):
        # 构建完整的文件路径
        path = os.path.join(directory, item)
        # 检查项是否是一个文件（而不是子目录）
        if os.path.isfile(path):
            filenames.append(item)  # 添加文件名到列表中
    return filenames


data_path = './data/Cuff-Less Blood Pressure Estimation/Samples'
csv_files = get_filenames_in_directory(data_path)

for csv_file in tqdm(csv_files):
    csv_path = os.path.join(data_path, csv_file)

    data = pd.read_csv(csv_path, header=None)
    ppg_arr = data.loc[0, :]
    abp_arr = data.loc[1, :]

    n_sequences = len(ppg_arr) // 1024

    ppg_matrix = np.empty((n_sequences, 1024), dtype=np.float64)
    abp_matrix = np.empty((n_sequences, 1024), dtype=np.float64)

    # 填充数组
    for i in range(n_sequences):
        start = i * 1024
        end = start + 1024
        ppg_matrix[i, :] = ppg_arr[start:end].values
        abp_matrix[i, :] = abp_arr[start:end].values


    ppg_df = pd.DataFrame(ppg_matrix)
    abp_df = pd.DataFrame(abp_matrix)
    mode = 'a' if os.path.exists('ppg_matrix.csv') else 'w'

    # 保存为CSV文件，使用追加模式
    ppg_df.to_csv('ppg_matrix.csv', mode=mode, index=False, header=None if mode == 'a' else True)
    abp_df.to_csv('abp_matrix.csv', mode=mode, index=False, header=None if mode == 'a' else True)



