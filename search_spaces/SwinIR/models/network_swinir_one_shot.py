# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

from email import header
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from optimizers.optim_factory import get_mixop, get_sampler
import numpy as np


def count_parameters(param_list):
    return np.sum(np.prod(v.size()) for v in param_list)


class LayerNormSampled(torch.nn.Module):

    def __init__(self, norm, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = norm

    def forward(self, x):
        weight = self.layer.weight[:self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        x = F.layer_norm(x[:, :, :self.emb_choice], (self.emb_choice, ),
                         weight=weight,
                         bias=bias,
                         eps=self.layer.eps)
        if x.shape[-1] != self.super_emb:
            x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)

        return x

    def get_parameters(self):
        weight = self.layer.weight[:self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        params_count = count_parameters([weight, bias])
        return params_count


class LinearSampled(torch.nn.Module):

    def __init__(self, layer, emb_choice, mlp_choice, super_emb):
        super().__init__()
        self.in_dim = emb_choice
        self.out_dim = emb_choice * mlp_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x):
        #print(x.shape)
        weight = self.layer.weight[:self.out_dim, :self.in_dim]
        bias = self.layer.bias[:self.out_dim]
        x = F.linear(x[:, :, :self.in_dim], weight=weight, bias=bias)
        out = torch.zeros([x.shape[0], x.shape[1], self.super_emb],
                          device=x.device)
        #print(x.shape)
        out[:, :, :x.shape[-1]] = x
        return out

    def get_parameters(self):
        weight = self.layer.weight[:self.out_dim, :self.in_dim]
        bias = self.layer.bias[:self.out_dim]
        params_count = count_parameters([weight, bias])
        return params_count


class LinearSampledReverse(torch.nn.Module):

    def __init__(self, layer, emb_choice, mlp_choice, super_emb):
        super().__init__()
        self.in_dim = emb_choice * mlp_choice
        self.out_dim = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x):
        #print(x.shape)
        weight = self.layer.weight[:self.out_dim, :self.in_dim]
        bias = self.layer.bias[:self.out_dim]
        x = F.linear(x[:, :, :self.in_dim], weight=weight, bias=bias)
        out = torch.zeros([x.shape[0], x.shape[1], self.super_emb],
                          device=x.device)
        #print(x.shape)
        out[:, :, :x.shape[-1]] = x
        return out

    def get_parameters(self):
        weight = self.layer.weight[:self.out_dim, :self.in_dim]
        bias = self.layer.bias[:self.out_dim]
        params_count = count_parameters([weight, bias])
        return params_count


class Mlp(nn.Module):

    def __init__(self,
                 mixop,
                 config,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  #mlp embed
        self.super_embed_dim = max(config["embed_dim"])
        self.super_mlp_ratio = max(config["mlp_ratio"])
        self.mixop = mixop
        self.fc1_op_list = []
        for e in config["embed_dim"]:
            for r in config["mlp_ratio"]:
                self.fc1_op_list.append(
                    LinearSampled(self.fc1, e, r,
                                  self.super_embed_dim * self.super_mlp_ratio))
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  #mlp embed
        self.fc2_op_list = []
        for e in config["embed_dim"]:
            for r in config["mlp_ratio"]:
                self.fc2_op_list.append(
                    LinearSampledReverse(self.fc2, e, r, self.super_embed_dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x, weights_emb_ratio):
        x = self.mixop.forward(x,
                               weights_emb_ratio,
                               self.fc1_op_list,
                               combi=True)
        x = self.act(x)
        x = self.drop(x)
        x = self.mixop.forward(x,
                               weights_emb_ratio,
                               self.fc2_op_list,
                               combi=True)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LinearSampled_qkv(torch.nn.Module):

    def __init__(self, layer, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x):
        weight = self.layer.weight[:3 * self.emb_choice, :self.emb_choice]
        bias = self.layer.bias[:3 * self.emb_choice]
        x = F.linear(x[:, :, :self.emb_choice], weight=weight, bias=bias)
        out = torch.zeros([x.shape[0], x.shape[1], 3 * self.super_emb],
                          device=x.device)
        out[:, :, :x.shape[-1]] = x
        return out

    def get_parameters(self):
        weight = self.layer.weight[:3 * self.emb_choice, :self.emb_choice]
        bias = self.layer.bias[:3 * self.emb_choice]
        params_count = count_parameters([weight, bias])
        return params_count


class LinearSampled_proj(torch.nn.Module):

    def __init__(self, layer, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x, discretize=False):
        weight = self.layer.weight[:self.emb_choice, :self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        x = F.linear(x[:, :, :self.emb_choice], weight=weight, bias=bias)
        out = torch.zeros([x.shape[0], x.shape[1], self.super_emb],
                          device=x.device)
        out[:, :, :x.shape[-1]] = x
        return out

    def get_parameters(self):
        weight = self.layer.weight[:self.emb_choice, :self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        params_count = count_parameters([weight, bias])
        return params_count


class HeadsOp(torch.nn.Module):

    def __init__(self, head_choice, emb_choice, super_emb, qk_scale,
                 relative_position_bias_table, relative_position_index,
                 attn_drop, window_size):
        super().__init__()
        self.head_choice = head_choice
        self.emb_choice = emb_choice
        self.head_dim = self.emb_choice // head_choice
        self.scale = qk_scale or self.head_dim**-0.5
        self.super_emb = super_emb
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = relative_position_bias_table
        self.relative_position_index = relative_position_index
        self.attn_drop = attn_drop
        self.window_size = window_size

    def forward(self, qkv_out, mask, B, N):
        qkv = qkv_out[:, :, :self.emb_choice * 3].reshape(
            B, N, 3, self.head_choice,
            self.emb_choice // self.head_choice).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[:, :self.head_choice][
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.head_choice, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.head_choice, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.emb_choice)
        out = torch.zeros(x.shape[0],
                          x.shape[1],
                          self.super_emb,
                          device=x.device)
        #print(x.shape)
        out[:, :, :x.shape[-1]] = x
        return out

    def get_parameters(self):
        relative_position_bias = self.relative_position_bias_table[:, :self.
                                                                   head_choice]
        params_count = count_parameters([relative_position_bias])
        return params_count


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 mixop,
                 config,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mixop = mixop
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_op_list = [
            LinearSampled_qkv(self.qkv, e, self.dim)
            for e in config["embed_dim"]
        ]
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv_op_list_head = []
        for e in config["embed_dim"]:
            for h in config["num_heads"]:
                self.qkv_op_list_head.append(
                    HeadsOp(h, e, dim, qk_scale,
                            self.relative_position_bias_table,
                            relative_position_index, self.attn_drop,
                            self.window_size))

        self.proj = nn.Linear(dim, dim)
        self.proj_op_list = [
            LinearSampled_proj(self.proj, e, self.dim)
            for e in config["embed_dim"]
        ]
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, weights_embed_dim, weights_heads, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.mixop.forward(x, weights_embed_dim, self.qkv_op_list)
        x, p1 = self.mixop.forward_swin_attn(
            qkv, [weights_embed_dim, weights_heads],
            self.qkv_op_list_head,
            mask,
            B_,
            N,
            add_params=True,
            combi=True)
        x, p2 = self.mixop.forward(x,
                                   weights_embed_dim,
                                   self.proj_op_list,
                                   add_params=True)
        x = self.proj_drop(x)
        return x, p1 + p2

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 mixop,
                 config,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.config = config
        self.mixop = mixop
        self.mask_list = [
            EmbedMask(e, self.dim) for e in self.config["embed_dim"]
        ]
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.super_embed_dim = max(self.config["embed_dim"])
        self.norm1 = norm_layer(dim)  #embed dim
        self.norm1_op_list = [
            LayerNormSampled(self.norm1, e, self.super_embed_dim)
            for e in self.config["embed_dim"]
        ]
        self.attn = WindowAttention(mixop,
                                    config,
                                    dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)  # embed_dim, num_heads

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  #embed_dim
        self.norm2_op_list = [
            LayerNormSampled(self.norm2, e, self.super_embed_dim)
            for e in self.config["embed_dim"]
        ]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(mixop,
                       config,
                       in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)  #num_head, mlp_ratio

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-100.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, weights_embed_dim, weights_mlp_ratio,
                weights_heads):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x, p1 = self.mixop.forward(x,
                                   weights_embed_dim,
                                   self.norm1_op_list,
                                   add_params=True)

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, p = self.attn(
                x_windows,
                weights_embed_dim,
                weights_heads,
                mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows, p = self.attn(x_windows,
                                        weights_embed_dim,
                                        weights_heads,
                                        mask=self.calculate_mask(x_size).to(
                                            x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x_temp, p1 = self.mixop.forward(shortcut,
                                        weights_embed_dim,
                                        self.mask_list,
                                        add_params=True)
        x = x_temp + self.drop_path(x)
        x2, p2 = self.mixop.forward(x,
                                    weights_embed_dim,
                                    self.norm2_op_list,
                                    add_params=True)
        x = x + self.drop_path(
            self.mlp(x2, [weights_embed_dim, weights_mlp_ratio]))

        return x, p1 + p2 + p

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 mixop,
                 id,
                 config,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.id = id
        self.mixop = mixop
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.config = config
        self.num_swin = self.config["num_swin"]
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(mixop,
                                 config,
                                 dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if
                                 (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)
        ])  #depth_sub
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, weights_num_swin, weights_embed_dim,
                weights_mlp_ratio, weights_heads):
        x_list = []
        p_list = []
        num_params = 0
        i = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x, p = blk(x, x_size, weights_embed_dim, weights_mlp_ratio[i],
                           weights_heads[i])
                num_params += p
                if i + 1 in self.num_swin:
                    x_list.append(x)
                    p_list.append(num_params)
            i = i + 1
        x, p = self.mixop.forward_depth(x_list,
                                        weights_num_swin[self.id],
                                        params_list=p_list,
                                        add_params=True)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, p

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 mixop,
                 id,
                 config,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.config = config
        self.mixop = mixop
        self.residual_group = BasicLayer(
            mixop,
            id,
            config,
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)  # embed_dim, num_heads, mlp_ratio

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
            self.conv_op_list = [
                ConvSampled(self.conv, e, dim)
                for e in self.config["embed_dim"]
            ]
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=0,
                                      embed_dim=dim,
                                      norm_layer=None)

        self.patch_unembed = PatchUnEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=0,
                                          embed_dim=dim,
                                          norm_layer=None)

    def forward(self, x, x_size, weights_emb, weights_num_swin,
                weights_mlp_ratio, weights_heads):
        resi_out, p1 = self.residual_group(x, x_size, weights_num_swin,
                                           weights_emb, weights_mlp_ratio,
                                           weights_heads)
        x2, p2 = self.mixop.forward(self.patch_unembed(resi_out, x_size),
                                    weights_emb,
                                    self.conv_op_list,
                                    add_params=True)
        x = self.patch_embed(x2) + x
        return x, p1 + p2

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0],
                                   x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self,
                 mixop,
                 config,
                 scale,
                 num_feat,
                 num_out_ch,
                 input_resolution=None):
        super().__init__()
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        self.config = config
        self.conv_pre = nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1)
        self.conv_pre_op_list = [
            ConvSampledIn(self.conv_pre, e, num_feat)
            for e in self.config["embed_dim"]
        ]
        self.mixop = mixop
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x, weights):
        return self.shuffle(
            self.mixop.forward(x, weights, self.conv_pre_op_list))

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class EmbedMask(nn.Module):

    def __init__(self, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb

    def forward(self, x):
        mask = torch.zeros([self.super_emb], device=x.device)
        mask[:self.emb_choice] = 1
        return x * mask

    def get_parameters(self):
        return 0


class EmbedMaskUnEmbed(nn.Module):

    def __init__(self, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb

    def forward(self, x, discretize):
        mask = torch.zeros([self.super_emb], device=x.device)
        mask[:self.emb_choice] = 1
        out = x.permute(0, 2, 3, 1) * mask
        return out.permute(0, 3, 1, 2)

    def get_parameters(self):
        return 0


class LayerNormSampled(torch.nn.Module):

    def __init__(self, layer, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x):
        weight = self.layer.weight[:self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        x = F.layer_norm(x[:, :, :self.emb_choice], (self.emb_choice, ),
                         weight=weight,
                         bias=bias,
                         eps=self.layer.eps)
        if x.shape[-1] != self.super_emb:
            x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)

        return x

    def get_parameters(self):
        weight = self.layer.weight[:self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        params_count = count_parameters([weight, bias])
        return params_count


class ConvSampled(torch.nn.Module):

    def __init__(self, layer, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x):
        weight = self.layer.weight[:self.emb_choice, :self.emb_choice, :, :]
        bias = self.layer.bias[:self.emb_choice]
        x = F.conv2d(x[:, :self.emb_choice, :, :],
                     weight=weight,
                     bias=bias,
                     stride=self.layer.stride,
                     padding=self.layer.padding)
        if x.shape[-1] != self.super_emb:
            out = torch.zeros(
                [x.shape[0], self.super_emb, x.shape[2], x.shape[3]],
                device=x.device)
            out[:, :self.emb_choice, :, :] = x
        return out

    def get_parameters(self):
        weight = self.layer.weight[:self.emb_choice, :self.emb_choice, :, :]
        bias = self.layer.bias[:self.emb_choice]
        params_count = count_parameters([weight, bias])
        return params_count


class ConvSampledIn(torch.nn.Module):

    def __init__(self, layer, emb_choice, super_emb):
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x):
        weight = self.layer.weight[:, :self.emb_choice, :, :]
        bias = self.layer.bias
        x = F.conv2d(x[:, :self.emb_choice, :, :],
                     weight=weight,
                     bias=bias,
                     stride=self.layer.stride,
                     padding=self.layer.padding)
        return x

    def get_parameters(self):
        weight = self.layer.weight[:, :self.emb_choice, :, :]
        bias = self.layer.bias
        params_count = count_parameters([weight, bias])
        return params_count


class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 mixop,
                 config,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.config = config
        num_feat = 64
        self.mixop = get_mixop(mixop)
        self.sampler = get_sampler(mixop)
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.super_embed_dim = max(self.config["embed_dim"])
        self.optimizer = mixop
        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_mask_list = [
            EmbedMask(e, self.super_embed_dim)
            for e in self.config["embed_dim"]
        ]
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  #embed_dim
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  #embed_dim

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))  #embed dim
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)  #embed dim?

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  #num RSTB search (layers)
            layer = RSTB(
                self.mixop,
                i_layer,
                config,
                dim=embed_dim,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]
                                  ):sum(depths[:i_layer +
                                               1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.num_rstb = self.config["num_rstb"]
        self.norm = norm_layer(self.num_features)  # embed dim
        self.norm_op_list = [
            LayerNormSampled(self.norm, e, self.super_embed_dim)
            for e in self.config["embed_dim"]
        ]
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1,
                                             1)  #embed dim
            self.conv_op_list = [
                ConvSampled(self.conv_after_body, e, self.super_embed_dim)
                for e in self.config["embed_dim"]
            ]
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))  #embed dim

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))  #embed dim
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                self.mixop, config, upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1]))  #embed dim
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))  #embed dim
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  #embed dim
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  #embed dim
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  #embed dim
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1,
                                       1)  #embed dim
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1,
                                       1)  #embed dim

        self.apply(self._init_weights)
        self._initialize_alphas()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _initialize_alphas(self):

        self.alphas_embed_dim = torch.nn.Parameter(
            1e-3 * torch.randn(1, len(self.config["embed_dim"])),
            requires_grad=True)
        self.alphas_mlp_ratio = torch.nn.Parameter(1e-3 * torch.randn(
            max(self.config["num_rstb"]), max(self.config["num_swin"]),
            len(self.config["mlp_ratio"])),
                                                   requires_grad=True)
        self.alphas_num_heads = torch.nn.Parameter(1e-3 * torch.randn(
            max(self.config["num_rstb"]), max(self.config["num_swin"]),
            len(self.config["num_heads"])),
                                                   requires_grad=True)
        self.alphas_num_rstb = torch.nn.Parameter(
            1e-3 * torch.randn(1, len(self.config["num_rstb"])),
            requires_grad=True)
        self.alphas_num_swin = torch.nn.Parameter(1e-3 * torch.randn(
            max(self.config["num_rstb"]), len(self.config["num_swin"])),
                                                  requires_grad=True)
        self._arch_parameters = [
            self.alphas_embed_dim, self.alphas_mlp_ratio,
            self.alphas_num_heads, self.alphas_num_rstb, self.alphas_num_swin
        ]

    def arch_parameters(self):
        return self._arch_parameters

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size -
                     h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size -
                     w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, weights_embed_dim, weights_mlp_ratio,
                         weights_num_heads, weights_num_rstb,
                         weights_num_swin):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x, p1 = self.mixop.forward(x, weights_embed_dim[0],
                                   self.conv_mask_list, True)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  #check
        i = 0
        x_list = []
        p_rstb = 0
        p_list = []
        for layer in self.layers:
            x, p = layer(x, x_size, weights_embed_dim[0], weights_num_swin,
                         weights_mlp_ratio[i], weights_num_heads[i])
            p_rstb += p
            if i + 1 in self.num_rstb:
                x_list.append(x)
                p_list.append(p_rstb)
            i = i + 1
        x, p2 = self.mixop.forward_depth(x_list,
                                         weights_num_rstb[0],
                                         params_list=p_list,
                                         add_params=True)
        x, p3 = self.mixop.forward(x, weights_embed_dim[0], self.norm_op_list,
                                   True)  # B L C
        x = self.patch_unembed(x, x_size)

        return x, p1 + p2 + p3

    def get_best_config(self):
        config = {}
        config["embed_dim"] = self.config["embed_dim"][torch.argmax(
            self.alphas_embed_dim, dim=-1)]
        config["num_rstb"] = self.config["num_rstb"][torch.argmax(
            self.alphas_num_rstb, dim=-1)]
        for d in range(4):
            #print("Block",d+1)
            #print("Number of Swins",self.config["num_swin"][torch.argmax(self.alphas_num_swin[d],dim=-1)])
            config["num_swin_" +
                   str(d)] = self.config["num_swin"][torch.argmax(
                       self.alphas_num_swin[d], dim=-1)]
            for i in range(6):
                config["num_heads_" + str(d) + "_" +
                       str(i)] = self.config["num_heads"][torch.argmax(
                           self.alphas_num_heads[d][i], dim=-1)]
                config["mlp_ratio_" + str(d) + "_" +
                       str(i)] = self.config["mlp_ratio"][torch.argmax(
                           self.alphas_mlp_ratio[d][i], dim=-1)]
        return config

    def forward(self, x, arch_params=None, tau=0.1):
        #self.tau = torch.Tensor([tau])
        if arch_params == None:
            arch_parameters_sampled = self.sampler.sample_step(
                self._arch_parameters)
        else:
            arch_parameters_sampled = arch_params
        #print(len(arch_parameters_sampled))
        weights_embed_dim = arch_parameters_sampled[0]
        weights_mlp_ratio = arch_parameters_sampled[1]
        weights_num_heads = arch_parameters_sampled[2]
        weights_num_rstb = arch_parameters_sampled[3]
        weights_num_swin = arch_parameters_sampled[4]
        #print("Embed dim", weights_embed_dim)
        #print("Mlp_ratio", weights_mlp_ratio)
        #print("Num heads", weights_num_heads)
        #print("Num rstb", weights_num_rstb)
        #print("Num swin", weights_num_swin)
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(
                x)) + x  #What to do about skips in general?
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            #print(x.shape)
            x = self.conv_first(x)
            #print(x.shape)
            x, p1 = self.forward_features(x, weights_embed_dim,
                                          weights_mlp_ratio, weights_num_heads,
                                          weights_num_rstb, weights_num_swin)
            x_part2, p3 = self.mixop.forward(x.permute(0, 2, 3, 1),
                                             weights_embed_dim[0],
                                             self.conv_mask_list,
                                             add_params=True)
            x_part2 = x_part2.permute(0, 3, 1, 2)
            x_part1, p2 = self.mixop.forward(x, weights_embed_dim[0],
                                             self.conv_op_list, True)
            x = x_part1 + x_part2
            x = self.upsample(x, weights_embed_dim[0])
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x,
                                                    scale_factor=2,
                                                    mode='nearest')))
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x,
                                                    scale_factor=2,
                                                    mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(
                self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale], p1 + p2 + p3

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


'''if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (64 // upscale // window_size + 1) * window_size
    width = (64 // upscale // window_size + 1) * window_size
    config = {"embed_dim":[60,96,108],"mlp_ratio": [1,2],"num_rstb" : [4,6],"num_heads" : [4,6],"num_swin" : [4,5,6
    ]}
    model = SwinIR("gdas",config,upscale=4, img_size=(64,64),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=108, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').cuda()
    print(height, width, model.flops() / 1e9)
    x = torch.randn((16, 3, height, width)).cuda()
    x,p = model(x,torch.Tensor([0.1]))
    print(x.shape)
    print(p)'''
