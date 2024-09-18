import torch
import torch.nn as nn
import numpy as np
from model.kan import KAN
from model.quantizer import VectorQuantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size):
        super(Encoder, self).__init__()

        # CNN part
        self.conv = nn.Conv1d(input_channels, conv_out_channels, kernel_size, padding=(kernel_size // 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # CNN forward
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()

        # LSTM part
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(conv_out_channels, hidden_size, num_layers, batch_first=True).to(device)

        # Fully connected layer
        # self.fc = nn.Linear(hidden_size, output_size)
        self.kan = KAN([hidden_size, 128, output_size]).to(device)

    def forward(self, x):
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


class VQVAE(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size, 
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)  # torch.Size([64, 64, 512])

        # 通过卷积层将编码器输出转换为嵌入维度
        # z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)

        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((64, 1, 1024))
    x = torch.tensor(x).float()

    input_channels = 1  # For example, 1 for univariate time series data
    conv_out_channels = 64  # Number of output channels for the convolution layer
    kernel_size = 3  # Size of the convolution kernel
    hidden_size = 128  # Hidden size for LSTM
    num_layers = 2  # Number of LSTM layers
    output_size = 1024  # Output size (e.g., length of the output vector)
    n_embeddings = 512
    embedding_dim = 64
    beta = 0.25

    # test encoder
    model = VQVAE(input_channels, conv_out_channels, kernel_size, hidden_size, num_layers, output_size, n_embeddings,
                  embedding_dim, beta)
    embedding_loss, x_hat, perplexity = model(x)
    print('x_hat shape:', x_hat.shape)

