import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from data_process.CustomDataset import CustomDataset
from model.patchtst2.patchTST import PatchTST
from data_process.DataModule import DataModule2
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2()
print("Load data done!")

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--context_points', type=int, default=1024, help='sequence length')
parser.add_argument('--target_points', type=int, default=1024, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=800, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
parser.add_argument('--c_in', type=int, default=2, help='num of channels')


args = parser.parse_args()
print('args:', args)

# 参数定义
num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1
# Initialize model
model = PatchTST(c_in=args.c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                res_attention=False
                ).to(device)
# 超参数
num_epochs = args.n_epochs_pretrain
learning_rate = args.lr

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import numpy as np


def mse_loss(y_true, y_pred):
    error = y_pred - y_true

    squared_error = error ** 2

    mse = np.mean(squared_error)

    return mse


model.load_state_dict(torch.load('model/param/patchtst_epoch800_240919_ppgecg.pth'))

model.eval()  # 设置模型为评估模式
total_loss = 0
inverse_loss = 0
with torch.no_grad():  # 在评估过程中不需要计算梯度
    for batch_inputs, batch_targets in test_dataloader:
        batch_inputs = batch_inputs.view(batch_inputs.shape[0], -1, args.c_in, args.patch_len)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        # 前向传播
        outputs = model(batch_inputs)
        outputs = outputs.squeeze(1)

        # 计算损失
        loss = criterion(outputs, batch_targets)
        total_loss += loss.item()

        outputs_inver = output_scaler.inverse_transform(outputs.cpu())
        batch_targets_inver = output_scaler.inverse_transform(batch_targets.cpu())
        inverse_loss += mse_loss(outputs_inver, batch_targets_inver)

        # batch_inputs = batch_inputs.view(batch_inputs.shape[0], -1)
        # outputs = outputs.view(batch_inputs.shape[0], -1)
        # show_image(batch_inputs, batch_targets, outputs)

average_loss = total_loss / len(test_dataloader)
rmse_loss = torch.sqrt(torch.tensor(average_loss))
inverse_loss = inverse_loss / len(test_dataloader)
print(f"average_loss: {average_loss:.4f}")
print(f"RMSE Loss: {rmse_loss.item():.4f}")
print(f"归一化前: {inverse_loss.item():.4f}")