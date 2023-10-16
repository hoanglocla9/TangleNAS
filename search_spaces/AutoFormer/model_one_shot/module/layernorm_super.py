import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
class LayerNormSuper(torch.nn.LayerNorm):

    def __init__(self, super_embed_dim: int) -> None:
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return None


class LayerNormSub(torch.nn.Module):

    def __init__(self,
                 layer: torch.nn.LayerNorm,
                 embed_choice: int,
                 super_emb: int,
                 pre_norm: bool,
                 before: bool = False,
                 after: bool = False) -> None:
        super(LayerNormSub, self).__init__()
        self.sampled_in_dim = embed_choice
        self.super_embed_dim = super_emb
        self.normalize_before = pre_norm
        self.before = before
        self.after = after
        self.layer = layer

    def maybe_layer_norm(self,
                         norm: torch.nn.LayerNorm,
                         x: torch.Tensor,
                         before: bool = False,
                         after: bool = False) -> torch.Tensor:
        weight = norm.weight[:self.sampled_in_dim]
        bias = norm.bias[:self.sampled_in_dim]
        assert before ^ after
        if after ^ self.normalize_before:
            return F.layer_norm(x, (self.sampled_in_dim, ),
                                weight=weight,
                                bias=bias,
                                eps=norm.eps)
        else:
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maybe_layer_norm(self.layer,
                                  x[:, :, :self.sampled_in_dim],
                                  before=self.before,
                                  after=self.after)
        if x.shape[-1] != self.super_embed_dim:
            x = F.pad(x, (0, self.super_embed_dim - x.shape[-1]), "constant",
                      0)
        return x

class LayerNormMixture(torch.nn.Module):

    def __init__(self,
                 layer: torch.nn.LayerNorm,
                 emb_choice_list: list,
                 pre_norm: bool,
                 before: bool = False,
                 after: bool = False) -> None:
        super(LayerNormMixture, self).__init__()
        self.emb_choice_list = emb_choice_list
        self.max_emb = max(emb_choice_list)
        self.normalize_before = pre_norm
        self.before = before
        self.after = after
        self.layer = layer

    def maybe_layer_norm(self,
                         weight: torch.Tensor,
                         bias: torch.Tensor,
                         x: torch.Tensor,
                         before: bool = False,
                         after: bool = False,
                         emb_choice: int = 192,
                         use_argmax=False) -> torch.Tensor:
        assert before ^ after
        if use_argmax == False:
            emb_choice = self.max_emb
        if after ^ self.normalize_before:
            return F.layer_norm(x[:, :, :emb_choice], (emb_choice, ),
                                weight=weight,
                                bias=bias,
                                eps=self.layer.eps)
        else:
            return x

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size = self.emb_choice_list[idx]
        start = 0 
        end = start + emb_size
        #print(op_weight.shape)
        weight_curr = alpha * op_weight[:end]
        if use_argmax == False:
            conv_weight += F.pad(weight_curr, (0,self.max_emb-emb_size), "constant", 0)
        else:
            conv_weight = weight_curr
        
        if op_bias is not None:
            bias = alpha * op_bias[:emb_size]
            if use_argmax == False:
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
        weight, bias = self.compute_weight_and_bias_mixture(weights, self.layer.weight, self.layer.bias, use_argmax=use_argmax)
        selected_emb = self.emb_choice_list[np.array([w.item() for w in weights]).argmax()]
        x = self.maybe_layer_norm(weight,
                                  bias,
                                  x,
                                  before=self.before,
                                  after=self.after,
                                  emb_choice=selected_emb,
                                  use_argmax=use_argmax)
        return x
    
# test layer norm mixture
'''if __name__ == "__main__":
    layer = torch.nn.LayerNorm(10)
    layer.weight.data = torch.ones_like(layer.weight.data)
    layer.bias.data = torch.ones_like(layer.bias.data)
    emb_choice_list = [2, 4, 6, 10]
    layer_norm_mixture = LayerNormMixture(layer, emb_choice_list, pre_norm=True, before=True, after=False)
    x = torch.randn(1, 1, 10)
    weights = torch.tensor([0.1, 0.4, 0.3, 0.2],requires_grad=True)
    y = layer_norm_mixture(x, weights, use_argmax=False)
    print(y.shape)
    print(y)
    loss = y.sum()
    loss.backward()
    print(weights.grad)'''
