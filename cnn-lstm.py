import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from data_process.CustomDataset import CustomDataset
from model.CNNLSTMModel import CNNLSTMModel
from data_process.DataModule import DataModule
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_scaler, output_scaler, train_dataloader, test_dataloader = DataModule()
# 参数定义
input_channels = 1        # For example, 1 for univariate time series data
conv_out_channels = 64    # Number of output channels for the convolution layer
kernel_size = 3           # Size of the convolution kernel
hidden_size = 128         # Hidden size for LSTM
num_layers = 2            # Number of LSTM layers
output_size = 1024        # Output size (e.g., length of the output vector)
seq_length = 1024         # Length of the input sequence

# Initialize model
model = CNNLSTMModel(input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size, seq_length).to(device)
# 超参数
# num_epochs = 800
# learning_rate = 0.001

criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def mse_loss(y_true, y_pred):  
    error = y_pred - y_true  

    squared_error = error ** 2  

    mse = np.mean(squared_error)  
      
    return mse  

model.load_state_dict(torch.load('model/param/cnn_lstm.pth'))

model.eval()  # 设置模型为评估模式
total_loss = 0
inverse_loss = 0
with torch.no_grad():  # 在评估过程中不需要计算梯度
    for batch_inputs, batch_targets in test_dataloader:
        # 将输入数据和目标输出移动到设备
        batch_inputs = batch_inputs.unsqueeze(1).to(device)  # 添加通道维度
        batch_targets = batch_targets.to(device)
        
        # 前向传播
        outputs = model(batch_inputs)
        
        # 计算损失
        loss = criterion(outputs, batch_targets)
        total_loss += loss.item()

        outputs_inver = output_scaler.inverse_transform(outputs.cpu())
        batch_targets_inver = output_scaler.inverse_transform(batch_targets.cpu())
        inverse_loss += mse_loss(outputs_inver, batch_targets_inver)

        
average_loss = total_loss / len(test_dataloader)
rmse_loss = torch.sqrt(torch.tensor(average_loss)) 
inverse_loss = inverse_loss / len(test_dataloader)
print(f"average_loss: {average_loss:.4f}")
print(f"RMSE Loss: {rmse_loss.item():.4f}")
print(f"归一化前: {inverse_loss.item():.4f}")