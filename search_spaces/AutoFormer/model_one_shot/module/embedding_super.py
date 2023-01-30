import torch
import torch.nn as nn
import torch.nn.functional as F
from model_one_shot.utils import to_2tuple
import numpy as np
from model_one_shot.utils import trunc_normal_


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
