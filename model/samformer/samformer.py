import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils.attention import scaled_dot_product_attention
from .utils.dataset import LabeledDataset
from .utils.revin import RevIN
from .utils.sam import SAM


class SAMFormerArchitecture(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.revin = RevIN(num_features=args.num_channels)
        self.compute_keys = nn.Linear(args.seq_len, args.hid_dim)
        self.compute_queries = nn.Linear(args.seq_len, args.hid_dim)
        self.compute_values = nn.Linear(args.seq_len, args.seq_len)
        self.linear_forecaster = nn.Linear(args.seq_len, args.seq_len)
        self.use_revin = args.use_revin
        self.pro = nn.Linear(args.seq_len * 2, args.seq_len, bias=True)

    def forward(self, x, flatten_output=True):
        # RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2) # (n, D, L)
        else:
            x_norm = x
        # Channel-Wise Attention
        queries = self.compute_queries(x_norm) # (n, D, hid_dim)
        keys = self.compute_keys(x_norm) # (n, D, hid_dim)
        values = self.compute_values(x_norm) # (n, D, L)
        if hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        out = x_norm + att_score # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out) # (n, D, H)
        # RevIN Denormalization
        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2) # (n, D, H)
        if flatten_output:
            out = out.reshape([out.shape[0], out.shape[1]*out.shape[2]])
            out = self.pro(out)
            return out
        else:
            return out

