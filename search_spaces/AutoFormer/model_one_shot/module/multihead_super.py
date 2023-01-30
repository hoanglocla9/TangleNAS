import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .Linear_super import LinearSuper
from .qkv_super import qkv_super, qkv_super_sampled, qkv_super_sampled_true
from optimizers.mixop.base_mixop import MixOp
from model_one_shot.utils import trunc_normal_


def softmax(x: torch.Tensor,
            dim: int,
            onnx_trace: bool = False) -> torch.Tensor:
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class LinearSampled(torch.nn.Module):

    def __init__(self, layer: torch.nn.Linear, emb_choice: int,
                 super_emb: int) -> None:
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        weight = self.layer.weight[:self.emb_choice, :]
        bias = self.layer.bias[:self.emb_choice]
        x = F.linear(x, weight=weight, bias=bias)
        #print(x.shape)
        if x.shape[-1] != self.super_emb:
            x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)

        return x


class HeadMaskAttn(nn.Module):

    def __init__(self, head_choice: int, relative_position: int,
                 super_emb_dim: int, rel_pos_embed_k: nn.Module,
                 rel_pos_embed_v: nn.Module, attn_drop: torch.nn.Dropout,
                 change_qkv: bool, max_heads: int) -> None:
        super(HeadMaskAttn, self).__init__()
        self.head_choice = head_choice
        self.relative_position = relative_position
        self.super_emb_dim = super_emb_dim
        self.rel_pos_embed_k = rel_pos_embed_k
        self.rel_pos_embed_v = rel_pos_embed_v
        self.attn_drop = attn_drop
        self.change_qkv = change_qkv
        self.sample_scale = (64)**-0.5
        self.max_heads = max_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, :self.head_choice * 64 * 3]
        B, N, _ = x.shape
        qkv = x.reshape(B, N, 3, self.head_choice, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.sample_scale
        if self.change_qkv:
            self.rel_pos_embed_k.set_sample_config(64)
        else:
            self.rel_pos_embed_k.set_sample_config(self.super_emb_dim //
                                                   self.head_choice)
        attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.head_choice * B, -1) @ self.rel_pos_embed_k(N, N).transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.head_choice, N, N) * self.sample_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        if self.change_qkv:
            self.rel_pos_embed_v.set_sample_config(64)
        else:
            self.rel_pos_embed_v.set_sample_config(self.super_emb_dim //
                                                   self.head_choice)
        #r_p_v = r_p_v_op(N, N)
        #print(attn.permute(2, 0, 1, 3).shape)
        attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.head_choice, -1)
        x = x + (attn_1 @ self.rel_pos_embed_v(N, N)).transpose(1, 0).reshape(
            B, self.head_choice, N, -1).transpose(2, 1).reshape(B, N, -1)
        #print(x.shape)
        if x.shape[-1] != self.max_heads * 64:
            x = F.pad(x, (0, int(self.max_heads * 64) - x.shape[-1]),
                      "constant", 0)
        return x


class RelativePosition2D_super(nn.Module):

    def __init__(self, num_units: int, max_relative_position: int) -> None:
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None

    def set_sample_config(self, sample_head_dim: int) -> None:
        self.sample_head_dim = sample_head_dim
        self.sample_embeddings_table_h = self.embeddings_table_h[:, :
                                                                 sample_head_dim]
        self.sample_embeddings_table_v = self.embeddings_table_v[:, :
                                                                 sample_head_dim]

    def calc_sampled_param_num(self) -> float:
        return self.sample_embeddings_table_h.numel(
        ) + self.sample_embeddings_table_v.numel()

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        # compute the row and column distance
        distance_mat_v = (range_vec_k[None, :] // int(length_q**0.5) -
                          range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (range_vec_k[None, :] % int(length_q**0.5) -
                          range_vec_q[:, None] % int(length_q**0.5))
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              "constant", 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.sample_embeddings_table_v[
            final_mat_v] + self.sample_embeddings_table_h[final_mat_h]

        return embeddings


class AttentionSuper(nn.Module):

    def __init__(self,
                 mixop: MixOp,
                 config: dict,
                 super_embed_dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: bool = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 normalization: bool = False,
                 relative_position: bool = False,
                 num_patches: int = None,
                 max_relative_position: int = 14,
                 scale: bool = False,
                 change_qkv: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.super_embed_dim = super_embed_dim
        self.fc_scale = scale
        self.mixop = mixop
        self.change_qkv = change_qkv
        self.config = config
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim,
                                 3 * 64 * max(self.config["num_heads"]),
                                 bias=qkv_bias)
            self.qkv_op_choice = []
            for e in self.config["embed_dim"]:
                for h in self.config["num_heads"]:
                    self.qkv_op_choice.append(
                        qkv_super_sampled_true(self.qkv, e, h,
                                               max(self.config["num_heads"])))

        else:
            self.qkv = LinearSuper(super_embed_dim,
                                   3 * super_embed_dim,
                                   bias=qkv_bias)
            self.qkv_op_choice = [
                qkv_super_sampled(self.qkv, e, self.super_embed_dim)
                for e in self.config["embed_dim"]
            ]
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position = relative_position
        #if self.relative_position:
        if self.change_qkv == True:
            self.rel_pos_embed_k = RelativePosition2D_super(
                64, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(
                64, max_relative_position)
        else:
            self.rel_pos_embed_k = RelativePosition2D_super(
                super_embed_dim // min(self.config["num_heads"]),
                max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(
                super_embed_dim // min(self.config["num_heads"]),
                max_relative_position)
        self.mask_head_op_list = [
            HeadMaskAttn(h, relative_position, self.super_embed_dim,
                         self.rel_pos_embed_k, self.rel_pos_embed_v,
                         self.attn_drop, self.change_qkv,
                         max(self.config["num_heads"]))
            for h in self.config["num_heads"]
        ]
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None
        if self.change_qkv:
            self.proj = LinearSuper(64 * max(self.config["num_heads"]),
                                    super_embed_dim)
            self.mask_emb_op_list = [
                LinearSampled(self.proj, e, super_embed_dim)
                for e in self.config["embed_dim"]
            ]
        else:
            self.proj = LinearSuper(super_embed_dim, super_embed_dim)
            self.mask_emb_op_list = [
                LinearSampled(self.proj, e, super_embed_dim)
                for e in self.config["embed_dim"]
            ]

    def forward(self, x: torch.Tensor, id: int,
                weights_embed_dim: torch.Tensor,
                weights_num_heads: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if self.change_qkv != True:
            x = self.mixop.forward(x, weights_embed_dim[0], self.qkv_op_choice)
        else:
            weights = [weights_embed_dim[0], weights_num_heads[id]]
            x = self.mixop.forward(x, weights, self.qkv_op_choice, combi=True)
        x = self.mixop.forward(x, weights_num_heads[id],
                               self.mask_head_op_list)
        x = self.mixop.forward(x, weights_embed_dim[0], self.mask_emb_op_list)
        x = self.proj_drop(x)
        return x
