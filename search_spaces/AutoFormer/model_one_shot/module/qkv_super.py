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

import itertools
class qkv_super_mixture_true(nn.Module):

    def __init__(self, layer: torch.nn.Linear, embed_choice_list: list, heads_choice_list: list):
        super().__init__()

        # super_in_dim and super_out_dim indicate the largest network!
        self.embed_choice_list = embed_choice_list
        self.max_emb = max(self.embed_choice_list)
        self.heads_choice_list = heads_choice_list
        self.emb_head_choice_list = list(itertools.product(self.embed_choice_list, self.heads_choice_list))
        self.out_dim_max = max(self.heads_choice_list) * 64 * 3
        self.layer = layer

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size, head_choice = self.emb_head_choice_list[idx]
        out_dim = head_choice*64*3
        start = 0 
        end = start + emb_size
        #print(op_weight.shape)
        #print(op_weight.shape)
        if not use_argmax:
            weight_curr = alpha*op_weight[:,:end]
            weight_curr = torch.cat([weight_curr[i:out_dim:3, :] for i in range(3)], dim=0)
            weight_curr = F.pad(weight_curr, (0,self.max_emb-emb_size,0,self.out_dim_max-out_dim))
        else:
            weight_curr = alpha*op_weight[:,:end]
            weight_curr = torch.cat([weight_curr[i:out_dim:3, :] for i in range(3)], dim=0)
            #print(weight_curr.shape)
        #print(weight_curr.shape)
        conv_weight = conv_weight+weight_curr
        if op_bias is not None:
            bias = alpha*op_bias[:out_dim]
            if not use_argmax:
                bias = F.pad(bias, (0,self.out_dim_max-out_dim), "constant", 0)

            conv_bias += bias

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
        #return F.linear(
        #    x[:, :, :self.embed_choice], self.layer.weight[:, :self.embed_choice],
        #    self.layer.bias)  #* (self.sample_scale if self.scale else 1)
        weight, bias = self.compute_weight_and_bias_mixture(weights, self.layer.weight, self.layer.bias, use_argmax=use_argmax)
        #print(weight)
        ##print(bias)
        #print(weight.shape)
        #print(bias.shape)
        argmax = np.array([w.item() for w in weights]).argmax()
        emb_size, head_choice = self.emb_head_choice_list[argmax]
        if use_argmax:
            out = F.linear(x[:,:,:emb_size], weight, bias)
        else:
            out = F.linear(x, weight, bias)
        #print(out.shape)
        return out
    
# test qkv super sampled
'''if __name__ == "__main__":
    linear_layer = torch.nn.Linear(30, 64*3*3)
    linear_layer.weight.data = torch.ones_like(linear_layer.weight.data)
    linear_layer.bias.data = torch.ones_like(linear_layer.bias.data)
    qkv_mixture = qkv_super_mixture_true(linear_layer, [18,24,30], [1, 2, 3])
    x = torch.ones(1, 2, 30)
    qkv_out = qkv_mixture(x, torch.tensor([0.1, 0.2, 0.9, 0.8, 0.5, 0.6, 0.7, 0.8, 0.4]), use_argmax=False)
    print(qkv_out.shape)
    print(qkv_out)
    qkv = qkv_out.reshape(1,2,3,3,-1).permute(2,0,3,1,4)
    print(qkv[0])
    print(qkv[1])
    print(qkv[2])'''