import torch
import torch.nn as nn
import torch.optim as optim
from model.patchtst2.patchTST import PatchTST
from data_process.DataModule import DataModule2
from datetime import datetime
from model.softs.SOFTS import SOFTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_inputs = batch_inputs.permute(0, 2, 1)

        outputs = model(batch_inputs)
        outputs = outputs.squeeze(2)

        loss = criterion(outputs, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"model/param/softs_{current_time}_epoch{num_epochs}.pth"
torch.save(model.state_dict(), file_name)
print("The model has been saved successfully!")