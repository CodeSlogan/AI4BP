import torch
import torch.nn as nn
import torch.optim as optim
from data_process.DataModule import DataModule2
from datetime import datetime
from model.moderntcn.ModernTCN import Model
from model.moderntcn.utils.str2bool import str2bool
from utils.eval_func import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser(description='ModernTCN')

# random seed
parser.add_argument('--random_seed', type=int, default=2025, help='random seed')


# forecasting task
parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1024, help='prediction sequence length')

#ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=8, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
parser.add_argument('--large_size', nargs='+',type=int, default=[51,51,51,51], help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
parser.add_argument('--dims', nargs='+',type=int, default=[128,128,128,128], help='dmodels in each stage')
parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256])

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=False, help='use_multi_scale fusion')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=800, help='train epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
print('args:', args)

input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2(args)
print("Load data done!")

# Initialize model
model = Model(args).to(device)
# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
# 超参数
num_epochs = args.train_epochs
learning_rate = args.learning_rate

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

is_train = True
if is_train:
    # 训练模型
    for epoch in range(num_epochs):
        print(f"{epoch}/{num_epochs}")
        step = 0
        model.train()
        for batch_inputs, batch_targets in train_dataloader:
            step += 1
            if step % 10 == 0:
                print(f"epoch:{epoch}, {step}/{len(train_dataloader)}")

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_inputs = batch_inputs.permute(0, 2, 1)

            outputs = model(batch_inputs)
            # outputs = outputs.squeeze(2)

            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        if epoch and epoch % 100 == 0:
            file_name = f"model/param/moderntcn_{loss}_epoch{epoch}.pth"
            torch.save(model.state_dict(), file_name)
            print(f"{file_name} has saved succesfully!")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"model/param/moderntcn_{current_time}_epoch{num_epochs}.pth"
    torch.save(model.state_dict(), file_name)
    print("The model has been saved successfully!")
else:
    model.load_state_dict(torch.load('model/param/moderntcn_20241114_001436_epoch800.pth'))

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
