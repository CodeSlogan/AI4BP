import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from .CustomDataset import CustomDataset
import os 

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