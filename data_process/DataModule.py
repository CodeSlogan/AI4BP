import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from .CustomDataset import CustomDataset, SequenceDataset
import scipy.io as sio
import numpy as np

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
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(normalized_inputs, normalized_outputs, test_size=0.3,
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

def DataModule2():
    seq_len = 1024
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mat_file = './data/data2/Part_1.mat'
    data = sio.loadmat(mat_file)

    ppgs = []
    abps = []
    ecgs = []

    for i in range(len(data['p'][0])):
        ppg = data['p'][0][i][0]
        abp = data['p'][0][i][1]
        ecg = data['p'][0][i][2]

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

    input1_scaler, input2_scaler, output_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

    ppgs = input1_scaler.fit_transform(ppgs)
    ecgs = input2_scaler.fit_transform(ecgs)
    abps = output_scaler.fit_transform(abps)

    train_input1, test_input1, train_input2, test_input2, train_output, test_output = train_test_split(
        ppgs, ecgs, abps, test_size=0.3, random_state=42)

    train_dataset = SequenceDataset(train_input1, train_input2, train_output)
    test_dataset = SequenceDataset(test_input1, test_input2, test_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return input1_scaler, input2_scaler, output_scaler, train_loader, test_loader


if __name__ == '__main__':
    DataModule2()
