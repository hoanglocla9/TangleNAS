import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from search_spaces.AutoFormer.model_one_shot.module.linear_super import LinearSuper
from  search_spaces.AutoFormer.model_one_shot.module.qkv_super import qkv_super, qkv_super_sampled, qkv_super_sampled_true, qkv_super_mixture_true
from optimizers.mixop.base_mixop import MixOp
from search_spaces.AutoFormer.model_one_shot.utils import trunc_normal_
from optimizers.mixop.entangle import EntangledOp
import itertools

import numpy as np
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

class LinearMixture(torch.nn.Module):

    def __init__(self, layer: torch.nn.Linear, emb_choice_list: list, head_choice_list: list) -> None:
        super().__init__()
        self.emb_choice_list = emb_choice_list
        self.head_choice_list = head_choice_list
        self.max_head = max(head_choice_list)
        self.layer = layer
        self.max_emb = max(emb_choice_list)
        # cross of emb choice and head choice
        self.head_emb_choice = list(itertools.product(emb_choice_list, head_choice_list))

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size,head_choice = self.head_emb_choice[idx]
        start = 0 
        end = start + emb_size
        #print(op_weight.shape)
        #print("original weight shape", op_weight.shape)
        weight_curr = alpha * op_weight[:end,:head_choice*64]
        if not use_argmax:
            conv_weight += F.pad(weight_curr, (0,(self.max_head*64)-(weight_curr.shape[-1]),0,self.max_emb-emb_size), "constant", 0)
        else:
            conv_weight += weight_curr
        
        if op_bias is not None:
            bias = alpha * op_bias[:emb_size]
            if not use_argmax:
                conv_bias += F.pad(bias,(0,self.max_emb-emb_size),"constant",0)
            else:
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
        #print(x.shape)
        weight, bias = self.compute_weight_and_bias_mixture(weights,self.layer.weight, self.layer.bias, use_argmax=use_argmax)
        x = F.linear(x, weight=weight, bias=bias)

        return x

class HeadMaskAttnMixture(nn.Module):

    def __init__(self, head_choice_list: list, emb_dim_choice_list: list, 
                 relative_position: int, rel_pos_embed_k: nn.Module,
                 rel_pos_embed_v: nn.Module, attn_drop: torch.nn.Dropout,
                 change_qkv: bool) -> None:
        super(HeadMaskAttnMixture, self).__init__()
        self.head_choice_list  = head_choice_list
        self.relative_position = relative_position
        self.emb_dim_choice_list = emb_dim_choice_list
        self.rel_pos_embed_k = rel_pos_embed_k
        self.rel_pos_embed_v = rel_pos_embed_v
        self.attn_drop = attn_drop
        self.change_qkv = change_qkv
        self.sample_scale = (64)**-0.5
        self.max_heads = max(self.head_choice_list)

    def _get_qkv(self, x: torch.Tensor, weights: torch.Tensor, idx: int, q, k, v, use_argmax=False):
        head_choice = self.head_choice_list[idx]
        #x = x[:, :, :head_choice * 64 * 3]
        #print(torch.sum(x))
        
        #print(weights[idx])
        B, N, _ = x.shape
        if not use_argmax:
            x_temp = x.reshape(B, N, 3, self.max_heads, -1).permute(2, 0, 3, 1, 4)
        else:
            x_temp = x.reshape(B, N, 3, head_choice, -1).permute(2, 0, 3, 1, 4)
        #print("X first line sum", torch.sum(x[0]))
        B, N, _, _, _ = x_temp.shape
        qkv = x_temp
        #print("Q before shit",torch.sum(qkv[0]))
        if not use_argmax:
            #print(qkv[0].shape)
            q_idx = F.pad(weights[idx]*qkv[0][:,:head_choice,:,:], (0,0,0,0,0,self.max_heads-head_choice,0,0),"constant", 0)
            k_idx = F.pad(weights[idx]*qkv[1][:,:head_choice,:,:], (0,0,0,0,0,self.max_heads-head_choice,0,0),"constant", 0)
            v_idx = F.pad(weights[idx]*qkv[2][:,:head_choice,:,:], (0,0,0,0,0,self.max_heads-head_choice,0,0),"constant", 0)
        else:
            q_idx = weights[idx]*qkv[0][:,:head_choice,:,:]
            k_idx = weights[idx]*qkv[1][:,:head_choice,:,:]
            v_idx = weights[idx]*qkv[2][:,:head_choice,:,:]
        q, k, v = q+q_idx, k+k_idx, v+v_idx
        return q, k , v

    def _get_qkv_mixture(self, x: torch.Tensor, weights = torch.Tensor, use_argmax=False):
        q = 0
        k = 0
        v = 0
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            q, k, v = self._get_qkv(
                x,
                weights,
                argmax,
                q,
                k,
                v,
                use_argmax=use_argmax
            )
        else:
            for i, _ in enumerate(weights):
                q, k ,v = self._get_qkv(
                    x,
                    weights,
                    i,
                    q,
                    k,
                    v,
                    use_argmax=use_argmax

                )
        return q, k ,v

    def forward(self, x: torch.Tensor, weights, use_argmax=False) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self._get_qkv_mixture(x, weights, use_argmax=use_argmax)
        selected_head = self.head_choice_list[np.array([w.item() for w in weights]).argmax()]
        #print(q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.sample_scale
        #print(attn.shape)
        self.rel_pos_embed_k.set_sample_config(64)
        if not use_argmax:
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.max_heads * B, -1) @ self.rel_pos_embed_k(N, N).transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.max_heads, N, N) * self.sample_scale
        else:
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, selected_head * B, -1) @ self.rel_pos_embed_k(N, N).transpose(2, 1)) \
                .transpose(1, 0).reshape(B, selected_head, N, N) * self.sample_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        #print(x.shape)
        self.rel_pos_embed_v.set_sample_config(64)
        #r_p_v = r_p_v_op(N, N)
        #print(attn.permute(2, 0, 1, 3).shape)
        if not use_argmax:
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.max_heads, -1)
            x = x + (attn_1 @ self.rel_pos_embed_v(N, N)).transpose(1, 0).reshape(
            B, self.max_heads, N, -1).transpose(2, 1).reshape(B, N, -1)
            #print(x.shape)
        else:
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * selected_head, -1)
            x = x + (attn_1 @ self.rel_pos_embed_v(N, N)).transpose(1, 0).reshape(
            B, selected_head, N, -1).transpose(2, 1).reshape(B, N, -1)
            #print(x.shape)
        #print(x.shape)
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
        if self.change_qkv:
            self.sample_scale = (64)**-0.5
        else:
            self.head_dim = self.super_emb_dim // self.head_choice
            self.sample_scale = (self.super_emb_dim // self.head_choice)**-0.5
        self.max_heads = max_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.change_qkv:
            x = x[:, :, :self.head_choice * 64 * 3]
        else:
            x = x[:, :, :self.super_emb_dim * 3]
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
        if self.change_qkv:
            if x.shape[-1] != self.max_heads * 64:
                x = F.pad(x, (0, int(self.max_heads * 64) - x.shape[-1]),
                      "constant", 0)
        else:
            if x.shape[-1] != self.super_emb_dim:
                x = F.pad(x, (0, self.super_emb_dim - x.shape[-1]),
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
        distance_mat_v = (torch.div(range_vec_k[None, :], int(length_q**0.5), rounding_mode="floor") -
                          torch.div(range_vec_q[:, None],int(length_q**0.5), rounding_mode="floor"))
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
                 change_qkv: bool = False,
                 use_we_v2 = False) -> None:
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
            if use_we_v2:
                op = qkv_super_mixture_true(self.qkv,self.config["embed_dim"], self.config["num_heads"])
                self.qkv_op_choice = self._init_entangled_op_cross(op, self.config["embed_dim"], self.config["num_heads"], "embed_head")
            else:
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
        if use_we_v2:
            op = HeadMaskAttnMixture(self.config["num_heads"], self.config["embed_dim"], relative_position, self.rel_pos_embed_k, self.rel_pos_embed_v, self.attn_drop, self.change_qkv)
            self.mask_head_op_list = self._init_entangled_op(op, self.config["num_heads"], op_name="num_heads")
        else:
            self.mask_head_op_list = [
            HeadMaskAttn(h, relative_position, self.super_embed_dim,
                         self.rel_pos_embed_k, self.rel_pos_embed_v,
                         self.attn_drop, self.change_qkv,
                         max(self.config["num_heads"]))
            for h in self.config["num_heads"]]
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None
        if self.change_qkv:
            proj = LinearSuper(64 * max(self.config["num_heads"]),
                                    super_embed_dim)
            if use_we_v2:
                self.op_proj = LinearMixture(proj,self.config["embed_dim"],self.config["num_heads"])
                self.mask_emb_op_list = self._init_entangled_op_cross(self.op_proj, self.config["embed_dim"], self.config["num_heads"], "emb_dim")
            else:
                self.mask_emb_op_list = torch.nn.ModuleList([LinearSampled(proj, e, super_embed_dim) for e in self.config["embed_dim"]])
        else:
            proj = LinearSuper(super_embed_dim, super_embed_dim)
            self.mask_emb_op_list = torch.nn.ModuleList([
                LinearSampled(proj, e, super_embed_dim)
                for e in self.config["embed_dim"]
            ])

    def _init_entangled_op_cross(self, op, choices1, choices2, op_name):
        choices_cross = list(itertools.product(choices1, choices2))
        ops = [EntangledOp(op=None, name=op_name) for i,j in choices_cross[:-1]] + [EntangledOp(op=op,name=op_name)]
        return ops

    def _init_entangled_op(self, op, choices, op_name):
        ops = [EntangledOp(op=None, name=op_name) for i in choices[:-1]] + [EntangledOp(op=op,name=op_name)]
        return ops

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
        x = self.mixop.forward(x, [weights_embed_dim[0],weights_num_heads[id]], self.mask_emb_op_list, combi=True)
        x = self.proj_drop(x)
        return x

# test head mask attm mixture
'''if __name__ == "__main__":
    head_choice_list = [2,4,8]
    emb_choice_list = [16,24,32]
    relative_position = 30
    max_relative_position = 30
    rel_pos_embed_k = RelativePosition2D_super(
                64, max_relative_position)
    rel_pos_embed_v = RelativePosition2D_super(
                64, max_relative_position)
    attn_drop = torch.nn.Dropout(0.1)
    change_qkv = True
    x = torch.rand(1,2,64*8*3)
    head_attn_mixture = HeadMaskAttnMixture(head_choice_list, emb_choice_list, relative_position, rel_pos_embed_k, rel_pos_embed_v, attn_drop, change_qkv=True)
    weights = torch.tensor([0.7,0.2,0.1])
    x = head_attn_mixture(x, weights, use_argmax=True)
    print(x.shape)
    print(x)'''
