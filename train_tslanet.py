import torch
import torch.nn as nn
import torch.optim as optim
from data_process.DataModule import DataModule2
from datetime import datetime
from model.tslanet.tslanet import TSLANet
from model.moderntcn.utils.str2bool import str2bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse
parser = argparse.ArgumentParser("TSLANet")

# forecasting lengths
parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1024, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# optimization
parser.add_argument('--train_epochs', type=int, default=800, help='train epochs')
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
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
        # outputs = outputs.squeeze(2)

        loss = criterion(outputs, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    if epoch and epoch % 100 == 0:
        file_name = f"model/param/tslanet_{loss}_epoch{epoch}.pth"
        torch.save(model.state_dict(), file_name)
        print(f"{file_name} has saved succesfully!")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"model/param/tslanet_{current_time}_epoch{num_epochs}.pth"
torch.save(model.state_dict(), file_name)
print("The model has been saved successfully!")
