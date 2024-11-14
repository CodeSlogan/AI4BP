import torch
from data_process.DataModule import DataModule2
from model.tslanet.tslanet import TSLANet
from model.moderntcn.utils.str2bool import str2bool
import numpy as np
import torch.nn as nn
from utils.eval_func import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("TSLANet")

# forecasting lengths
parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1024, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# optimization
parser.add_argument('--train_epochs', type=int, default=800, help='train epochs')
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--seed', type=int, default=42)

# model
parser.add_argument('--emb_dim', type=int, default=64, help='dimension of model')
parser.add_argument('--depth', type=int, default=3, help='num of layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout value')
parser.add_argument('--patch_size', type=int, default=64, help='size of patches')
parser.add_argument('--mask_ratio', type=float, default=0.4)
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

# TSLANet components:
parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
parser.add_argument('--ICB', type=str2bool, default=True)
parser.add_argument('--ASB', type=str2bool, default=True)
parser.add_argument('--adaptive_filter', type=str2bool, default=True)


args = parser.parse_args()
print('args:', args)

input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2(args)
print("Load data done!")

# Initialize model
model = TSLANet(args).to(device)
# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

criterion = nn.MSELoss()

model.load_state_dict(torch.load('model/param/tslanet_20241113_235907_epoch800.pth'))

model.eval()  # 设置模型为评估模式
total_loss = 0
inverse_loss = 0
MAE_SBP, MSE_SBP, MAE_DBP, MSE_DBP = [], [], [], []
SD_SBP, SD_DBP = [], []
SBP5, SBP10, SBP15 = [], [], []
DBP5, DBP10, DBP15 = [], [], []

with (torch.no_grad()):  # 在评估过程中不需要计算梯度
    for batch_inputs, batch_targets in test_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_inputs = batch_inputs.permute(0, 2, 1)

        # 前向传播
        outputs = model(batch_inputs)

        # 计算损失
        loss = criterion(outputs, batch_targets)
        total_loss += loss.item()

        outputs_inver = output_scaler.inverse_transform(outputs.cpu())
        batch_targets_inver = output_scaler.inverse_transform(batch_targets.cpu())
        inverse_loss += mse_loss(outputs_inver, batch_targets_inver)

        mae_sbp, mse_sbp, mae_dbp, mse_dbp, sd_peaks, sd_troughs, peak_percentages, trough_percentages = calculate_batch_errors(outputs_inver, batch_targets_inver)
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

print(f"SBP5: {np.mean(SBP5)*100:.4f}%, SBP10: {np.mean(SBP10)*100:.4f}%, SBP15: {np.mean(SBP15)*100:.4f}%")
print(f"DBP5: {np.mean(DBP5)*100:.4f}%, DBP10: {np.mean(DBP10)*100:.4f}%, DBP15: {np.mean(DBP15)*100:.4f}%")