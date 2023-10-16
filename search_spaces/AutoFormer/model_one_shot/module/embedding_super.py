import torch
import torch.nn as nn
import torch.nn.functional as F
from search_spaces.AutoFormer.model_one_shot.utils import to_2tuple
import numpy as np
from search_spaces.AutoFormer.model_one_shot.utils import trunc_normal_


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


class PatchembedSuper(torch.nn.Module):  #TODO: Better name?

    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 2,
                 in_chans: int = 3,
                 embed_dim: int = 100,
                 scale: bool = False,
                 abs_pos: bool = True,
                 super_dropout: float = 0.,
                 pre_norm: bool = True) -> None:
        super(PatchembedSuper, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.super_dropout = super_dropout
        self.normalize_before = pre_norm
        self.super_embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans,
                              self.super_embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.super_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.super_embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        self.scale = scale

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #sample_embed_dim = self.emb_choice
        sampled_weight = self.proj.weight  #[:sample_embed_dim, ...]
        sampled_bias = self.proj.bias  #[:sample_embed_dim, ...]
        sampled_cls_token = self.cls_token  #[..., :sample_embed_dim]
        sampled_pos_embed = self.pos_embed  #[..., :sample_embed_dim]
        sample_dropout = 0  #calc_dropout(self.patch_emb_layer.super_dropout,
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x,
                     sampled_weight,
                     sampled_bias,
                     stride=self.patch_size,
                     padding=self.proj.padding,
                     dilation=self.proj.dilation).flatten(2).transpose(1, 2)
        #if self.patch_emb_layer.scale:
        #    return x * sampled_scale

        x = torch.cat((sampled_cls_token.expand(B, -1, -1), x), dim=1)
        if self.abs_pos:
            x = x + sampled_pos_embed
        x = F.dropout(x, p=sample_dropout, training=self.training)
        return x


class PatchembedMixture(torch.nn.Module):

    def __init__(self, layer: torch.nn.Linear, emb_choice_list: list) -> None:
        super(PatchembedMixture, self).__init__()
        self.emb_choice_list = emb_choice_list
        self.layer = layer
        self.max_emb = max(self.emb_choice_list)

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size = self.emb_choice_list[idx]
        start = 0 
        end = start + emb_size
        #print(op_weight.shape)
        weight_curr = alpha * op_weight[:end,:,:,:]
        if use_argmax == False:
            conv_weight +=F.pad(weight_curr, (0,0,0,0,0,0,0,self.max_emb-emb_size), "constant", 0)
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

    def _compute_embeds(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size = self.emb_choice_list[idx]
        start = 0 
        end = start + emb_size
        #print(op_weight.shape)
        weight_curr = alpha*op_weight[:,:,:end]
        if use_argmax == False:
            conv_weight += F.pad(weight_curr, (0,self.max_emb-emb_size,0,0,0,0), "constant", 0)
        else:
            conv_weight = weight_curr
        
        if op_bias is not None:
            bias = alpha * op_bias[:,:,:emb_size]
            if use_argmax == False:
                conv_bias += F.pad(bias,(0,self.max_emb-emb_size,0,0,0,0),"constant",0)
            else:
                conv_bias = bias

        return conv_weight, conv_bias

    def compute_embeds_mixture(self, weights, op_weight, op_bias, use_argmax=False):
        conv_weight = 0
        conv_bias = 0
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            conv_weight, conv_bias = self._compute_embeds(
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
                conv_weight, conv_bias = self._compute_embeds(
                    weights=weights,
                    idx=i,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    op_weight=op_weight,
                    op_bias=op_bias
                )
        if op_bias==None:
            conv_bias = op_bias
        return conv_weight, conv_bias

    def forward(self, x: torch.Tensor, weights: torch.Tensor, use_argmax=False) -> torch.Tensor:
        weight_proj, bias_proj = self.compute_weight_and_bias_mixture(weights, self.layer.proj.weight, self.layer.proj.bias, use_argmax=use_argmax)
        cls_token, pos_embed = self.compute_embeds_mixture(weights, self.layer.cls_token, self.layer.pos_embed, use_argmax=use_argmax)
        sample_dropout = calc_dropout(self.layer.super_dropout,
                                      self.layer.super_embed_dim,
                                      self.layer.super_embed_dim)
        if self.layer.scale:
            sampled_scale = 1
        B, C, H, W = x.shape
        assert H == self.layer.img_size[0] and W == self.layer.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.layer.img_size[0]}*{self.layer.img_size[1]})."
        x = F.conv2d(x,
                     weight_proj,
                     bias_proj,
                     stride=self.layer.patch_size,
                     padding=self.layer.proj.padding,
                     dilation=self.layer.proj.dilation).flatten(2).transpose(
                         1, 2)
        if self.layer.scale:
            return x * sampled_scale
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)
        if self.layer.abs_pos:
            x = x + pos_embed
        x = F.dropout(x, p=sample_dropout, training=self.training)
        return x

class PatchembedSub(torch.nn.Module):

    def __init__(self, layer: torch.nn.Linear, emb_choice: int) -> None:
        super(PatchembedSub, self).__init__()
        self.emb_choice = emb_choice
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_embed_dim = self.emb_choice
        sampled_weight = self.layer.proj.weight[:sample_embed_dim, ...]
        sampled_bias = self.layer.proj.bias[:sample_embed_dim, ...]
        sampled_cls_token = self.layer.cls_token[..., :sample_embed_dim]
        sampled_pos_embed = self.layer.pos_embed[..., :sample_embed_dim]
        sample_dropout = calc_dropout(self.layer.super_dropout,
                                      sample_embed_dim,
                                      self.layer.super_embed_dim)
        if self.layer.scale:
            sampled_scale = self.layer.super_embed_dim / sample_embed_dim
        B, C, H, W = x.shape
        assert H == self.layer.img_size[0] and W == self.layer.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.layer.img_size[0]}*{self.layer.img_size[1]})."
        x = F.conv2d(x,
                     sampled_weight,
                     sampled_bias,
                     stride=self.layer.patch_size,
                     padding=self.layer.proj.padding,
                     dilation=self.layer.proj.dilation).flatten(2).transpose(
                         1, 2)
        if self.layer.scale:
            return x * sampled_scale

        x = torch.cat((sampled_cls_token.expand(B, -1, -1), x), dim=1)
        if self.layer.abs_pos:
            x = x + sampled_pos_embed
        x = F.dropout(x, p=sample_dropout, training=self.training)
        if x.shape[-1] != self.layer.super_embed_dim:
            x = F.pad(x, (0, self.layer.super_embed_dim - x.shape[-1]),
                      "constant", 0)
        return x
# test patch embed mixture
'''if __name__ == "__main__":
    emb_layer = PatchembedSuper(embed_dim=10)
    emb_layer.proj.weight.data = torch.ones_like(emb_layer.proj.weight.data)
    emb_layer.proj.bias.data = torch.ones_like(emb_layer.proj.bias.data)
    emb_layer.cls_token.data = torch.ones_like(emb_layer.cls_token.data)
    emb_layer.pos_embed.data = torch.ones_like(emb_layer.pos_embed.data)
    patch_mixture = PatchembedMixture(emb_layer,[6,8,10])
    x = torch.ones(1,3,32,32)
    weights = torch.tensor([0.7,0.2,0.1],requires_grad=True)
    out = patch_mixture(x,weights,use_argmax=False)
    print(out.shape)
    print(out)
    loss = out.sum()
    loss.backward()
    print(weights.grad)'''