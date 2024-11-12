import torch
import torch.nn as nn
import torch.optim as optim
from data_process.DataModule import DataModule
from model.vqvae import VQVAE
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_scaler, output_scaler, train_dataloader, test_dataloader = DataModule()

# 参数定义
input_channels = 1  # For example, 1 for univariate time series data
conv_out_channels = 128  # Number of output channels for the convolution layer
kernel_size = 3  # Size of the convolution kernel
hidden_size = 256  # Hidden size for LSTM
num_layers = 2  # Number of LSTM layers
output_size = 1024  # Output size (e.g., length of the output vector)
seq_length = 1024  # Length of the input sequence
n_embeddings = 512
embedding_dim = 64
beta = 0.25

# Initialize model
model = VQVAE(input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size,
              n_embeddings, embedding_dim, beta).to(device)

# 超参数
num_epochs = 300
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
        embedding_loss, outputs, perplexity = model(batch)

        # 计算损失
        loss = criterion(outputs, batch_targets) + embedding_loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()  # 设置模型为评估模式
total_loss = 0
with torch.no_grad():  # 在评估过程中不需要计算梯度
    for batch_inputs, batch_targets in test_dataloader:
        # 将输入数据和目标输出移动到设备
        batch_inputs = batch_inputs.unsqueeze(1).to(device)  # 添加通道维度
        batch_targets = batch_targets.to(device)

        # 前向传播
        embedding_loss, outputs, perplexity = model(batch_inputs)

        # 计算损失
        loss = criterion(outputs, batch_targets)
        total_loss += loss.item()

average_loss = total_loss / len(test_dataloader)
print(f"average_loss: {average_loss:.4f}")

import matplotlib.pyplot as plt

y1 = output_scaler.inverse_transform(batch_targets.cpu())
y2 = output_scaler.inverse_transform(outputs.cpu())

for i in range(0, len(y1)):
    plt.figure()
    plt.plot(y1[i, :], color='red')
    plt.plot(y2[i, :])
    plt.show()
