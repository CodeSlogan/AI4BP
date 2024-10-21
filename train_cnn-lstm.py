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
from data_process.DataModule import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input_scaler, output_scaler, train_dataloader, test_dataloader = DataModule()
input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2(True)

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
num_epochs = 800
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for batch_inputs, batch_targets in train_dataloader:
        # 将输入数据移动到设备
        batch = batch_inputs.unsqueeze(1).to(device)  # 添加通道维度

        # 前向传播
        outputs = model(batch)

        # 计算损失
        loss = criterion(outputs, batch_targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'model/param/cnn_lstm.pth')


def show_image(batch_inputs, batch_targets, outputs):
    batch_inputs = batch_inputs.squeeze(1)
    x = input1_scaler.inverse_transform(batch_inputs.cpu())  # ppg
    y1 = output_scaler.inverse_transform(batch_targets.cpu())  # abp
    y2 = output_scaler.inverse_transform(outputs.cpu())  # predict

    for i in range(0, len(y1)):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(x[i, :], label='PPG')
        axs[0].set_title('PPG')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Value')
        axs[0].legend()

        axs[1].plot(y1[i, :], color='red', label='ABP')
        axs[1].plot(y2[i, :], color='g', label='Predict')
        axs[1].set_title('ABP and Predict')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Value')
        axs[1].legend()

        plt.tight_layout()
        plt.show()


import numpy as np


def mse_loss(y_true, y_pred):
    error = y_pred - y_true

    squared_error = error ** 2

    mse = np.mean(squared_error)

    return mse


def find_sbp_dbp(y):
    """
    Find sbp and dbp in a 1D sequence.
    """
    sbp = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    dbp = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
    return sbp, dbp


def calculate_batch_errors(preds, true):
    """
    Calculate MAE and MSE for sbp and dbp across a batch of sequences.
    """
    batch_size, seq_length = preds.shape
    mae_sbp = []
    mse_sbp = []
    mae_dbp = []
    mse_dbp = []
    all_peak_errors = []
    all_trough_errors = []

    for i in range(batch_size):
        sbp, dbp = find_sbp_dbp(true[i])
        sbp_errors = np.abs(preds[i, sbp] - true[i, sbp])
        dbp_errors = np.abs(preds[i, dbp] - true[i, dbp])
        all_peak_errors.extend(sbp_errors)
        all_trough_errors.extend(dbp_errors)

        mae_p, mse_p = np.mean(np.abs(preds[i, sbp] - true[i, sbp])), np.mean(
            (preds[i, sbp] - true[i, sbp]) ** 2)
        mae_t, mse_t = np.mean(np.abs(preds[i, dbp] - true[i, dbp])), np.mean(
            (preds[i, dbp] - true[i, dbp]) ** 2)
        mae_sbp.append(mae_p)
        mse_sbp.append(mse_p)
        mae_dbp.append(mae_t)
        mse_dbp.append(mse_t)

    sd_peaks = np.std(all_peak_errors) if all_peak_errors else np.nan
    sd_troughs = np.std(all_trough_errors) if all_trough_errors else np.nan

    thresholds = [5, 10, 15]
    peak_percentages = [np.mean(np.array(all_peak_errors) <= thresh) for thresh in thresholds]
    trough_percentages = [np.mean(np.array(all_trough_errors) <= thresh) for thresh in thresholds]

    return np.mean(mae_sbp), np.mean(mse_sbp), np.mean(mae_dbp), np.mean(
        mse_dbp), sd_peaks, sd_troughs, peak_percentages, trough_percentages


model.load_state_dict(torch.load('model/param/cnn_lstm.pth'))

model.eval()  # 设置模型为评估模式
total_loss = 0
inverse_loss = 0
MAE_SBP, MSE_SBP, MAE_DBP, MSE_DBP = [], [], [], []
SD_SBP, SD_DBP = [], []
SBP5, SBP10, SBP15 = [], [], []
DBP5, DBP10, DBP15 = [], [], []

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

        mae_sbp, mse_sbp, mae_dbp, mse_dbp, sd_peaks, sd_troughs, peak_percentages, trough_percentages = calculate_batch_errors(
            outputs_inver, batch_targets_inver)
        MAE_SBP.append(mae_sbp)
        MSE_SBP.append(mse_sbp)
        MAE_DBP.append(mae_dbp)
        MSE_DBP.append(mse_dbp)
        SD_SBP.append(sd_peaks)
        SD_DBP.append(sd_troughs)
        SBP5.append(peak_percentages[0])
        SBP10.append(peak_percentages[1])
        SBP15.append(peak_percentages[2])
        DBP5.append(trough_percentages[0])
        DBP10.append(trough_percentages[1])
        DBP15.append(trough_percentages[2])

average_loss = total_loss / len(test_dataloader)
rmse_loss = torch.sqrt(torch.tensor(average_loss))
inverse_loss = inverse_loss / len(test_dataloader)
print(f"average_loss: {average_loss:.4f}")
print(f"RMSE Loss: {rmse_loss.item():.4f}")
print(f"归一化前总loss: {inverse_loss.item():.4f}")
print(f"MAE SBP: {np.mean(MAE_SBP):.4f}")
print(f"MSE SBP: {np.mean(MSE_SBP):.4f}")
print(f"SD_SBP: {np.mean(SD_SBP):.4f}")
print(f"MAE DBP: {np.mean(MAE_DBP):.4f}")
print(f"MSE DBP: {np.mean(MSE_DBP):.4f}")
print(f"SD_DBP: {np.mean(SD_DBP):.4f}")

print(f"SBP5: {np.mean(SBP5) * 100:.4f}%, SBP10: {np.mean(SBP10) * 100:.4f}%, SBP15: {np.mean(SBP15) * 100:.4f}%")
print(f"DBP5: {np.mean(DBP5) * 100:.4f}%, DBP10: {np.mean(DBP10) * 100:.4f}%, DBP15: {np.mean(DBP15) * 100:.4f}%")

show_image(batch_inputs, batch_targets, outputs)
