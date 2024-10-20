import torch
import torch.nn as nn
import torch.optim as optim
from model.patchtst2.patchTST import PatchTST
from data_process.DataModule import DataModule2
from datetime import datetime
from model.cross_models.cross_former import Crossformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input1_scaler, input2_scaler, output_scaler, train_dataloader, test_dataloader = DataModule2()
print("Load data done!")

import argparse

parser = argparse.ArgumentParser(description='CrossFormer')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')
parser.add_argument('--in_len', type=int, default=1024, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=1024, help='output MTS length (\tau)')
parser.add_argument('--seg_len', type=int, default=16, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

parser.add_argument('--data_dim', type=int, default=2, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')

args = parser.parse_args()
print('args:', args)

# Initialize model
model = Crossformer(
            args.data_dim,
            args.in_len,
            args.out_len,
            args.seg_len,
            args.win_size,
            args.factor,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.dropout,
            args.baseline,
            device
        ).float().to(device)
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
file_name = f"model/param/crossformer_{current_time}_epoch{num_epochs}.pth"
torch.save(model.state_dict(), file_name)
print("The model has been saved successfully!")