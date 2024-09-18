import torch
import torch.nn as nn

from model.kan import KAN


class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size,
                 seq_length):
        super(CNNLSTMModel, self).__init__()

        # CNN part
        self.conv = nn.Conv1d(input_channels, conv_out_channels, kernel_size, padding=(kernel_size // 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculating the sequence length after convolution and pooling
        self.conv_seq_length = seq_length // 2

        # LSTM part
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(conv_out_channels, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        # self.fc = nn.Linear(hidden_size, output_size)
        self.kan = KAN([hidden_size, 128, output_size])

    def forward(self, x):
        # CNN forward
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        # Prepare data for LSTM
        x = x.permute(0, 2, 1)

        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))

        # Fully connected layer
        # out = self.fc(out[:, -1, :])
        out = self.kan(out[:, -1, :])

        return out