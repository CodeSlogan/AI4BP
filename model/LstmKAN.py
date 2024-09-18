import torch.nn as nn

from model.kan import KAN

class LstmKAN(nn.Module):  # 定义一个名为lstm的类，继承自nn.Module
    # 初始化函数，定义模型的各层和参数
    def __init__(self, input_size=8, hidden_size=32, num_layers=1, output_size=1, dropout=0, batch_first=True):
        super(LstmKAN, self).__init__()  # 调用父类的构造函数
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size  # 设置LSTM的隐藏层大小
        self.input_size = input_size  # 设置LSTM的输入特征维度
        self.num_layers = num_layers  # 设置LSTM的层数
        self.output_size = output_size  # 设置输出的维度
        self.dropout = dropout  # 设置Dropout概率
        self.batch_first = batch_first  # 设置batch_first参数，决定输入输出张量的维度顺序
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=self.batch_first, dropout=self.dropout)  # 定义LSTM层
        # self.linear = nn.Linear(self.hidden_size, self.output_size) # 定义线性层
        self.kan = KAN([self.hidden_size, 64, self.output_size])

    def forward(self, x):  # 前向传播函数
        # 通过LSTM层，得到输出out和隐藏状态hidden, cell
        out, (hidden, cell) = self.rnn(
            x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # print(f"hidden.shape: {hidden.shape}")
        # print(f"hidden[-1].shape: {hidden[-1].shape}")
        # out = self.linear(hidden.reshape(a * b, c))
        # out = self.linear(hidden) # 将hidden通过线性层
        out = self.kan(hidden[-1])
        return out  # 返回输出
