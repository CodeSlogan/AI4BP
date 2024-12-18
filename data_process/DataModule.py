import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from .CustomDataset import *
import scipy.io as sio
import numpy as np
from scipy.signal import butter, lfilter

def DataModule():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppg_pd = pd.read_csv('./data/ppg_matrix.csv')
    abp_pd = pd.read_csv('./data/abp_matrix.csv')
    # 生成示例数据
    inputs = torch.tensor(ppg_pd.values, dtype=torch.float)
    outputs = torch.tensor(abp_pd.values, dtype=torch.float)

    # 使用 MinMaxScaler 对输入和输出分别进行归一化
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    normalized_inputs = input_scaler.fit_transform(inputs)
    normalized_outputs = output_scaler.fit_transform(outputs)

    # 划分数据集，按7:3的比例
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(normalized_inputs, normalized_outputs,
                                                                              test_size=0.3,
                                                                              random_state=42)

    # 转换为Tensor
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    train_outputs = torch.tensor(train_outputs, dtype=torch.float32, device=device)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_outputs = torch.tensor(test_outputs, dtype=torch.float32, device=device)

    # 创建自定义数据集
    train_dataset = CustomDataset(train_inputs, train_outputs)
    test_dataset = CustomDataset(test_inputs, test_outputs)

    # 定义批次大小
    batch_size = 64  # 可以根据显存大小进行调整

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return input_scaler, output_scaler, train_dataloader, test_dataloader


def DataModule2(config, only_ppg=False):
    seq_len = config.seq_len
    batch_size = config.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = './data/data2/'
    all_ppgs = []
    all_abps = []
    all_ecgs = []
    tot_segments = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            mat_file = os.path.join(folder_path, file_name)
            data = sio.loadmat(mat_file)

            ppgs = []
            abps = []
            ecgs = []

            for i in range(len(data['p'][0])):
                ppg = data['p'][0][i][0]
                abp = data['p'][0][i][1]
                ecg = data['p'][0][i][2]

                num_segments = len(ppg) // seq_len
                tot_segments += num_segments

                for j in range(num_segments):
                    ppg_segment = np.array(ppg[j * seq_len:(j + 1) * seq_len])
                    abp_segment = np.array(abp[j * seq_len:(j + 1) * seq_len])
                    ecg_segment = np.array(ecg[j * seq_len:(j + 1) * seq_len])

                    # 检查是否存在nan值
                    if np.isnan(ppg_segment).any() or np.isnan(abp_segment).any() or np.isnan(ecg_segment).any():
                        continue
                    ppgs.append(ppg_segment)
                    ecgs.append(ecg_segment)
                    abps.append(abp_segment)

            # 将当前文件提取的数据合并到总的数据列表中
            all_ppgs.extend(ppgs)
            all_abps.extend(abps)
            all_ecgs.extend(ecgs)

    print(f"The {tot_segments} of {seq_len} length segments have been loaded!")
    ppgs = np.array(all_ppgs)
    abps = np.array(all_abps)
    ecgs = np.array(all_ecgs)

    if np.isnan(ppgs).any() or np.isnan(abps).any() or np.isnan(ecgs).any():
        raise ValueError("The all_ppgs, all_abps or all_ecgs contains nan values! Program terminated.")

    input1_scaler, input2_scaler, output_scaler = (
        MinMaxScaler(feature_range=(0, 1)), StandardScaler(), MinMaxScaler(feature_range=(0, 1)))

    ppgs = input1_scaler.fit_transform(ppgs)
    ecgs = input2_scaler.fit_transform(ecgs)
    abps = output_scaler.fit_transform(abps)

    tot_train_input1, test_input1, tot_train_input2, test_input2, tot_train_output, test_output = train_test_split(
        ppgs, ecgs, abps, test_size=0.2, random_state=2025)

    train_input1, val_input1, train_input2, val_input2, train_output, val_output = train_test_split(
        tot_train_input1, tot_train_input2, tot_train_output, test_size=0.25, random_state=2025)

    if not only_ppg:
        train_dataset = SequenceDataset(train_input1, train_input2, train_output)
        val_dataset = SequenceDataset(val_input1, val_input2, val_output)
        test_dataset = SequenceDataset(test_input1, test_input2, test_output)
    else:
        train_dataset = SequenceDatasetPPG(train_input1, train_input2, train_output)
        val_dataset = SequenceDatasetPPG(val_input1, val_input2, val_output)
        test_dataset = SequenceDatasetPPG(test_input1, test_input2, test_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return input1_scaler, input2_scaler, output_scaler, train_loader, val_loader, test_loader


def njuDataModule(config, only_ppg=False):
    seq_len = config.seq_len
    batch_size = config.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = './data/real_data/raw_data/'

    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_paths.append(os.path.join(root, file))

    data = pd.DataFrame()

    for file_path in file_paths:
        temp_data = pd.read_csv(file_path, header=0)
        temp_data.columns = ['ppg1', 'xxx', 'ecg', 'abp']
        data = pd.concat([data, temp_data], ignore_index=True)

    ppgs = []
    abps = []
    ecgs = []

    ppg = data['ppg1'].values.reshape(-1, 1)
    abp = data['abp'].values.reshape(-1, 1)
    ecg = data['ecg'].values.reshape(-1, 1)

    input1_scaler, input2_scaler, output_scaler = (
        MinMaxScaler(feature_range=(0, 1)), StandardScaler(), MinMaxScaler(feature_range=(0, 1)))
    ppg = input1_scaler.fit_transform(ppg).flatten()
    ecg = input2_scaler.fit_transform(ecg).flatten()
    abp = output_scaler.fit_transform(abp).flatten()

    num_segments = len(ppg) // seq_len

    for j in range(num_segments):
        ppg_segment = ppg[j * seq_len:(j + 1) * seq_len]
        ppgs.append(np.array(ppg_segment))

        abp_segment = abp[j * seq_len:(j + 1) * seq_len]
        abps.append(np.array(abp_segment))

        ecg_segment = ecg[j * seq_len:(j + 1) * seq_len]
        ecgs.append(np.array(ecg_segment))

    ppgs = np.array(ppgs)
    abps = np.array(abps)
    ecgs = np.array(ecgs)

    train_input1, test_input1, train_input2, test_input2, train_output, test_output = train_test_split(
        ppgs, ecgs, abps, test_size=0.9, random_state=2025)

    if not only_ppg:
        train_dataset = SequenceDataset(train_input1, train_input2, train_output)
        test_dataset = SequenceDataset(test_input1, test_input2, test_output)
    else:
        train_dataset = SequenceDatasetPPG(train_input1, train_input2, train_output)
        test_dataset = SequenceDatasetPPG(test_input1, test_input2, test_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return input1_scaler, input2_scaler, output_scaler, train_loader, test_loader


if __name__ == '__main__':
    njuDataModule()
