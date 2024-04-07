import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

import itertools
class LinearSubRatioMixture(nn.Module):

    def __init__(self,
                 layer: torch.nn.Linear,
                 emb_choice_list: list,
                 ratio_choice_list: list,
                 reverse: bool = False) -> None:
        super(LinearSubRatioMixture, self).__init__()

        # super_in_dim and super_out_dim indicate the largest network!
        self.layer = layer
        self.emb_choice_list = emb_choice_list
        self.ratio_choice_list = ratio_choice_list
        self.emb_ratio_choice_list = list(itertools.product(self.emb_choice_list, self.ratio_choice_list))
        self.reverse = reverse
        self.max_dim = int(max(self.emb_choice_list)*max(self.ratio_choice_list))
        self.max_emb = max(self.emb_choice_list)
        #self.activation_fn = gelu
        self.layer = layer

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size, ratio = self.emb_ratio_choice_list[idx]

        #print(op_weight.shape)
        if not self.reverse:
            weight_curr = alpha * op_weight[:int(emb_size * ratio
                                            ), :emb_size]
            if not use_argmax:
                conv_weight += F.pad(weight_curr, (0,self.max_emb-emb_size,0,self.max_dim-int(emb_size * ratio)), "constant", 0)
            else:
                conv_weight = weight_curr
            if op_bias is not None:
                bias = alpha * op_bias[:int(emb_size * ratio)]
                if not use_argmax:
                    conv_bias += F.pad(bias,(0,self.max_dim-int(emb_size * ratio)),"constant",0)
                else:
                    conv_bias = bias

        else:
            weight_curr = alpha * op_weight[:emb_size, :int(emb_size * ratio
                                            ),]
            if not use_argmax:
                conv_weight += F.pad(weight_curr, (0,self.max_dim-int(emb_size * ratio),0,self.max_emb-emb_size), "constant", 0)
            else:
                conv_weight = weight_curr
            if op_bias is not None:
                bias = alpha * op_bias[:emb_size]
            if not use_argmax:
                conv_bias += F.pad(bias,(0,self.max_emb-emb_size),"constant",0)     
            else:
                conv_bias = bias    

        return conv_weight, conv_bias

    def compute_weight_and_bias_mixture(self, weights, op_weight, op_bias, use_argmax=False):
        conv_weight = 0
        conv_bias = 0
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            conv_weight, conv_bias = self._compute_weight_and_bias(
                weights=weights,
                idx=argmax,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                op_weight=op_weight,
                op_bias=op_bias,
                use_argmax=use_argmax
            )
        else:
            for i, _ in enumerate(weights):
                conv_weight, conv_bias = self._compute_weight_and_bias(
                    weights=weights,
                    idx=i,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    op_weight=op_weight,
                    op_bias=op_bias,
                    use_argmax=use_argmax
                )
        if op_bias==None:
            conv_bias = op_bias
        return conv_weight, conv_bias

    def forward(self, x: torch.Tensor, weights: torch.Tensor, use_argmax=False) -> torch.Tensor:
        weight, bias = self.compute_weight_and_bias_mixture(weights,self.layer.weight, self.layer.bias, use_argmax=use_argmax)
        #print(weight)
        #print(bias)
        argmax = np.array([w.item() for w in weights]).argmax()
        emb_size, ratio = self.emb_ratio_choice_list[argmax]
        if use_argmax:
            if not self.reverse:
                x = F.linear(x[:, :, :int(emb_size)], weight, bias)
            else:
                x = F.linear(x[:, :, :int(emb_size * ratio)], weight, bias)
        else:
            x = F.linear(x, weight, bias)
        return x
    
# test linear sub ratio mixture
'''if __name__ == "__main__":
    layer = torch.nn.Linear(12, 3)
    layer.weight.data = torch.ones_like(layer.weight.data)
    layer.bias.data = torch.ones_like(layer.bias.data)
    layer = LinearSubRatioMixture(layer, [1,2,3], [2,3,4], reverse=True)
    x = torch.ones((1,1,12))
    weights = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.9,0.7,0.8,0.6], requires_grad=True)
    y = layer(x, weights, use_argmax=False)
    print(y.shape)
    print(y)
    y.sum().backward()
    print(weights.grad)'''