import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LinearSuper(nn.Linear):

    def __init__(self,
                 super_in_dim: int,
                 super_out_dim: int,
                 bias: bool = True,
                 uniform_: callable = None,
                 non_linear: str = 'linear',
                 scale: bool = False) -> None:
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def _reset_parameters(self, bias: torch.Tensor, uniform_: callable,
                          non_linear: str) -> None:
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight,
                        self.bias)  #* (self.sample_scale if self.scale else 1)


class LinearSubRatioEmb(nn.Module):

    def __init__(self,
                 layer: torch.nn.Linear,
                 super_emb: int,
                 super_out_dim: int,
                 emb_choice: int,
                 ratio_choice: float,
                 reverse: bool = False) -> None:
        super(LinearSubRatioEmb, self).__init__()

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_emb = super_emb
        self.super_out_dim = super_out_dim
        self.reverse = reverse
        # input_dim and output_dim indicate the current sampled size
        self.emb_choice = emb_choice
        self.ratio_choice = ratio_choice
        #self.activation_fn = gelu
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.reverse:
            weight = self.layer.weight[:int(self.emb_choice * self.ratio_choice
                                            ), :self.emb_choice]
            bias = self.layer.bias[:int(self.emb_choice * self.ratio_choice)]
            x = F.linear(x[:, :, :self.emb_choice], weight, bias)
            if x.shape[-1] != self.super_out_dim:
                x = F.pad(x, (0, self.super_out_dim - x.shape[-1]), "constant",
                          0)
        else:
            weight = self.layer.weight[:self.
                                       emb_choice, :int(self.emb_choice *
                                                        self.ratio_choice)]
            bias = self.layer.bias[:self.emb_choice]
            x = F.linear(x[:, :, :int(self.emb_choice * self.ratio_choice)],
                         weight, bias)
            if x.shape[-1] != self.super_emb:
                x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)
        return x
