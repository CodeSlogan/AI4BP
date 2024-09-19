import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from data_process.CustomDataset import CustomDataset
from model.patchtst.patchTST import PatchTST
from data_process.DataModule import *
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input_scaler, output_scaler, train_dataloader, test_dataloader = DataModule()
input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2(True)
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


args = parser.parse_args()
print('args:', args)

# 参数定义
num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1 
# Initialize model
model = PatchTST(c_in=1,
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

# 训练模型
for epoch in range(num_epochs):
    print(f"{epoch}/{num_epochs}")
    step = 0
    model.train()
    for batch_inputs, batch_targets in train_dataloader:
        step += 1
        if step % 10 == 0:
            print(f"epoch:{epoch}, {step}/{len(train_dataloader)}")

        batch_inputs = batch_inputs.view(batch_inputs.shape[0], -1, 1, args.patch_len)
        batch = batch_inputs.to(device)  # 添加通道维度
        batch_targets = batch_targets.to(device)

        outputs = model(batch)
        outputs = outputs.view(batch_inputs.shape[0], -1)

        loss = criterion(outputs, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"model/param/patchtst_{current_time}_epoch{num_epochs}_onlyPPG.pth"
torch.save(model.state_dict(), file_name)
print("The model has been saved successfully!")