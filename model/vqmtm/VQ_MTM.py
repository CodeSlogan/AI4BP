import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Callable
from .layers.BERT_Enc import Trans_Conv
from .layers.Quantize import Quantize
from .layers.Embed import PositionalEmbedding

from typing import Optional

class VQ_MTM(nn.Module):
    def __init__(self, d_model: int, patch_size: int, dropout: float, in_channels: int, hidden_channels: int, mask_ratio: float,
                 num_layers: int, codebook_size: int, activation: str='gelu', linear_dropout: float=0.5,
                 mask_type: str='poisson', split_num: int=1, **kwargs):
        
        super(VQ_MTM, self).__init__(**kwargs)

        # Hyper-parameters
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.d_model = d_model
        self.split_num = split_num

        # instance norm
        self.instance_norm = nn.InstanceNorm1d(patch_size)

        # Project the patch_dim to the d_model dimension
        self.embed = nn.Linear(patch_size, d_model)
        self.activation_embed = self._get_activation_fn(activation)
        self.norm = nn.LayerNorm(d_model)
        
        # embed the time series to patch
        self.pos_embed = PositionalEmbedding(
            d_model=d_model,
            max_len=100
        )
        
        # Transformer Encoder
        self.TOKEN_CLS = torch.normal(mean=0, std=0.02, size=(1, 1, 1, d_model)).cuda()
        self.TOKEN_CLS.requires_grad = False
        self.register_buffer("CLS", self.TOKEN_CLS)
        
        self.encoder = nn.ModuleList([
            Trans_Conv(
                d_model=d_model, 
                dropout=dropout, 
                in_channels=in_channels, 
                out_channels=in_channels, 
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Different decoder for different down stream tasks
        self.decoder_cls = nn.Conv1d(in_channels=in_channels,
                                    out_channels=1,
                                    kernel_size=1)
        self.activation = self._get_activation_fn(activation)
        self.final_projector_cls = nn.Linear(d_model, 1024)
        
        self.linear_dropout = nn.Dropout(p=linear_dropout)
        

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        B, C, T = x.shape
        assert T % self.patch_size == 0, f"Time series length should be divisible by patch_size, not {T} % {self.patch_size}"
        x = x.view(B, C, -1, self.patch_size) # (B, C, T, patch_size)

        # Embedding
        y = self.embed(x) # (B, C, T, d_model)
        y = self.activation_embed(y)
        y = self.norm(y) # (B, C, T, d_model)

        # Add CLS token
        y = torch.concat([self.CLS.repeat(B, C, 1, 1), y], dim=2) # (B, C, T+1, d_model)

        y = y.view(-1, *y.shape[2:])
        pos_embed = self.pos_embed(y) # (B*C', T+1, d_model)
        y = y + pos_embed

        # Transformer Encoder
        y = y.view(B, -1, *y.shape[1:]) # (B, C', T+1, d_model)
        for encoder in self.encoder:
            y = encoder(y) # (B, C', T+1, d_model)
        
        # Decoder
        y = y[:, :, 0, :]
        y = self.decoder_cls(y).squeeze(1)
        y = self.activation(y)
        y = self.linear_dropout(y)
        y = self.final_projector_cls(y)
        return y


    @staticmethod
    def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    
    def random_masking(self, shape, mask_ratio, device):
        N, L = shape  # batch, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # create mask
        mask = torch.ones([N, L], dtype=bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :len_keep], 0)

        return mask
    

class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()
        self.args = args
        self.model = VQ_MTM(
            d_model=args.d_model,
            patch_size=args.freq,
            dropout=args.dropout,
            in_channels=args.num_nodes,
            hidden_channels=args.hidden_channels,
            num_layers=args.e_layers,
            codebook_size=args.codebook_item,
            activation=args.activation,
            # task_name=args.task_name,
            linear_dropout=args.linear_dropout,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
        )
    
    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
