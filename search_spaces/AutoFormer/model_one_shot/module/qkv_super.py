import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class qkv_super(nn.Linear):

    def __init__(self,
                 super_in_dim: int,
                 super_out_dim: int,
                 bias: bool = True,
                 uniform_: callable = None,
                 non_linear: str = 'linear',
                 scale: bool = False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode: bool = True) -> None:
        self.profiling = mode

    def _reset_parameters(self, bias: torch.Tensor, uniform_: callable,
                          non_linear: str) -> None:
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight,
                        self.bias)  #* (self.sample_scale if self.scale else 1)


class qkv_super_sampled(nn.Module):

    def __init__(self, layer: torch.nn.Linear, embed_choice: int,
                 super_embed_dim: int) -> None:
        super().__init__()

        # super_in_dim and super_out_dim indicate the largest network!
        self.embed_choice = embed_choice
        self.super_embed_dim = super_embed_dim
        self.layer = layer
        # input_dim and output_dim indicate the current sampled size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x[:, :, :self.embed_choice],
            self.layer.weight[:, :self.embed_choice],
            self.layer.bias)  #* (self.sample_scale if self.scale else 1)


class qkv_super_sampled_true(nn.Module):

    def __init__(self, layer: torch.nn.Linear, embed_choice: int,
                 num_heads: int, max_heads: int):
        super().__init__()

        # super_in_dim and super_out_dim indicate the largest network!
        self.embed_choice = embed_choice
        self.out_dim = num_heads * 64 * 3
        self.layer = layer
        self.max_heads = max_heads
        # input_dim and output_dim indicate the current sampled size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return F.linear(
        #    x[:, :, :self.embed_choice], self.layer.weight[:, :self.embed_choice],
        #    self.layer.bias)  #* (self.sample_scale if self.scale else 1)
        sample_weight = self.layer.weight[:, :self.embed_choice]
        sample_weight = torch.cat(
            [sample_weight[i:self.out_dim:3, :] for i in range(3)], dim=0)
        if self.layer.bias != None:
            bias = self.layer.bias[:self.out_dim]
        else:
            bias = None
        out = F.linear(x[:, :, :self.embed_choice], sample_weight, bias)
        #print(out.shape)
        #print(self.max_heads*64*3)
        if out.shape[-1] != self.max_heads * 64 * 3:
            out = F.pad(out, (0, int(self.max_heads * 64 * 3) - out.shape[-1]),
                        "constant", 0)
        #print(out.shape)
        return out
