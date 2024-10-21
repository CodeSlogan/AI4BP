import torch
import torch.nn as nn
import torch.optim as optim
from model.patchtst2.patchTST import PatchTST
from data_process.DataModule import DataModule2
from datetime import datetime
from model.softs.SOFTS import SOFTS
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return np.mean(mae_sbp), np.mean(mse_sbp), np.mean(mae_dbp), np.mean(mse_dbp), sd_peaks, sd_troughs, peak_percentages, trough_percentages

input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2()
print("Load data done!")

import argparse

parser = argparse.ArgumentParser(description='SOFTS')

# data loader
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# forecasting task
parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=1024, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# model define
# parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_core', type=int, default=512, help='dimension of core')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
parser.add_argument('--use_norm', type=int, default=False, help='use norm and denorm')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=800, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--save_model', action='store_true')

args = parser.parse_args()
print('args:', args)

# Initialize model
model = SOFTS(args).to(device)
# 超参数
num_epochs = args.train_epochs
learning_rate = args.learning_rate

criterion = nn.MSELoss()

def mse_loss(y_true, y_pred):
    error = y_pred - y_true

    squared_error = error ** 2

    mse = np.mean(squared_error)

    return mse


model.load_state_dict(torch.load('model/param/softs_20241021_150203_epoch800.pth'))

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
        outputs = outputs.squeeze(2)

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

