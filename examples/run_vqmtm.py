import torch
import torch.nn as nn
import torch.optim as optim
from data_process.DataModule import DataModule2
from datetime import datetime
from model.vqmtm.VQ_MTM import Model
from utils.eval_func import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("VQMTM")

parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
parser.add_argument('--freq', type=int, default=128, help='sample frequency')
parser.add_argument('--activation', type=str, default='gelu', help='activation function, options:[relu, gelu]')

# ssl task
parser.add_argument('--input_len', type=int, default=1024, help='input sequence length')
parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')
parser.add_argument('--output_len', type=int, default=1024, help='prediction sequence length')
parser.add_argument('--time_step_len', type=int, default=1, help='time step length')
parser.add_argument('--use_fft', action='store_true', help='use fft or not', default=False)
parser.add_argument('--loss_fn', type=str, default='mae', help='loss function, options:[mse, mae]')

# classification task
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')

# detection task
parser.add_argument('--scale_ratio', type=float, default=1.0, help='scale ratio of train data')
parser.add_argument('--balanced', action='store_true', help='balanced data or not', default=False)

# graph setting
parser.add_argument('--graph_type', type=str, default='correlation', help='graph type, option:[distance, correlation]')
parser.add_argument('--top_k', type=int, default=3, help='top k in graph or top k in TimesNet')
parser.add_argument('--directed', action='store_true', help='directed graph or not', default=False)
parser.add_argument('--filter_type', type=str, default='dual_random_walk', help='filter type')

# model define
parser.add_argument('--num_nodes',type=int, default=2, help='Number of nodes in graph.')
parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of RNN layers in encoder and/or decoder.')
parser.add_argument('--rnn_units', type=int, default=64, help='Number of hidden units in DCRNN.')
parser.add_argument('--dcgru_activation', type=str, choices=('relu', 'tanh'), default='tanh', help='Nonlinear activation used in DCGRU cells.')
parser.add_argument('--input_dim', type=int, default=None, help='Input seq feature dim.')
parser.add_argument('--output_dim', type=int, default=None, help='Output seq feature dim.')
parser.add_argument('--max_diffusion_step', type=int, default=2, help='Maximum diffusion step.')
parser.add_argument('--cl_decay_steps', type=int, default=3000, help='Scheduled sampling decay steps.')
parser.add_argument('--use_curriculum_learning', default=False, action='store_true', help='Whether to use curriculum training for seq-seq model.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability.')
parser.add_argument('--d_hidden', type=int, default=8, help='Hidden state dimension.')
parser.add_argument('--num_kernels', type=int, default=5, help='Number of each kind of kernel.')
parser.add_argument('--d_model', type=int, default=256, help='hidden dimension of channels')
parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers.')
parser.add_argument('--attn_head', type=int, default=4, help='Number of attention heads.')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size.')
parser.add_argument('--enc_type', type=str, default='rel', help='Encoder type, options:[abs, rel]')
parser.add_argument('--linear_dropout', type=float, default=0.5, help='linear dropout ratio')
parser.add_argument('--hidden_channels', type=int, default=16, help='hidden channels of conv')
parser.add_argument('--d_layers', type=int, default=3, help='Number of decoder layers.')
parser.add_argument('--global_pool', action='store_true', default=False, help='global pool or not')

# quantization
parser.add_argument('--codebook_item', type=int, default=1024, help='number of embedding vectors')
parser.add_argument('--codebook_num', type=int, default=4, help='number of codebooks')

# masking
parser.add_argument('--mask_ratio', type=float, default=0.2, help='mask ratio')
parser.add_argument('--mask_length', type=int, default=10, help='mask length')
parser.add_argument('--no_overlap', action='store_true', default=False, help='mask overlap or not')
parser.add_argument('--min_space', type=int, default=1, help='min space between mask')
parser.add_argument('--mask_dropout', type=float, default=0.0, help='mask dropout ratio')
parser.add_argument('--mask_type', type=str, default='poisson', help='mask type, options:[static, uniform, normal, poisson]')
parser.add_argument('--lm', type=int, default=3, help='lm of poisson mask')

# optimization
parser.add_argument('--train_epochs', type=int, default=800, help='train epochs')
parser.add_argument('--patience', type=int, default=0, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
parser.add_argument('--max_norm', type=float, default=1.0, help='max norm of grad')
parser.add_argument('--use_scheduler', action='store_true', default=False, help='use scheduler or not')
parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

# SimMTM
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of softmax')
parser.add_argument('--positive_nums', type=int, default=2, help='positive nums of contrastive learning')
parser.add_argument('--dimension', type=int, default=64, help='dimension of SimMTM')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

# log setting
parser.add_argument('--eval_every', type=int, default=1, help='evaluate every X epochs')
parser.add_argument('--adj_every', type=int, default=10, help='display adj matrix every X epochs')

# pretrain
parser.add_argument('--pretrained_path', type=str, default=None, help='pretrain model path')

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
            # batch_inputs = batch_inputs.permute(0, 2, 1)

            outputs = model(batch_inputs)
            # outputs = outputs.squeeze(2)

            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        if epoch and epoch % 60 == 0:
            file_name = f"model/param/vqmtm_{loss}_epoch{epoch}.pth"
            torch.save(model.state_dict(), file_name)
            print(f"{file_name} has saved succesfully!")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"model/param/vqmtm_{current_time}_epoch{num_epochs}.pth"
    torch.save(model.state_dict(), file_name)
    print("The model has been saved successfully!")
else:
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
            # batch_inputs = batch_inputs.permute(0, 2, 1)

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
