import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_one_shot.module.linear_super import LinearSuper, LinearSubRatioEmb, LinearSubRatioMixture
from model_one_shot.module.layernorm_super import LayerNormSuper, LayerNormSub, LayerNormMixture
from model_one_shot.module.multihead_super import AttentionSuper
from model_one_shot.module.embedding_super import PatchembedSuper, PatchembedSub, PatchembedMixture
from model_one_shot.utils import trunc_normal_
from model_one_shot.utils import DropPath
import itertools
import numpy as np
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from optimizers.mixop.base_mixop import MixOp
from optimizers.optim_factory import get_mixop, get_sampler
from optimizers.mixop.entangle import EntangledOp
from torch.autograd import Variable


class LayerNormSampled(torch.nn.Module):

    def __init__(self, layer: torch.nn.LayerNorm, emb_choice: int,
                 super_emb: int) -> None:
        super().__init__()
        self.layer = layer
        self.emb_choice = emb_choice
        self.super_emb = super_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.layer.weight[:self.emb_choice]
        bias = self.layer.bias[:self.emb_choice]
        x = F.layer_norm(x[:, :, :self.emb_choice], (self.emb_choice, ),
                         weight=weight,
                         bias=bias,
                         eps=self.layer.eps)
        if x.shape[-1] != self.super_emb:
            x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)

        return x


class LayerNormMainMixture(torch.nn.Module):

    def __init__(self, layer: torch.nn.LayerNorm, emb_choice_list: list) -> None:
        super().__init__()
        self.layer = layer
        self.emb_choice_list = emb_choice_list
        self.max_emb = max(emb_choice_list)

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size = self.emb_choice_list[idx]
        start = 0
        end = start + emb_size
        # print(op_weight.shape)
        weight_curr = alpha*op_weight[:end]
        if not use_argmax:
            conv_weight += F.pad(weight_curr, (0, self.max_emb-emb_size), "constant", 0)
        else:
            conv_weight += weight_curr

        if op_bias is not None:
            bias = alpha * op_bias[:emb_size]
            if not use_argmax:
                conv_bias += F.pad(bias, (0, self.max_emb-emb_size), "constant", 0)
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
        if op_bias == None:
            conv_bias = op_bias
        return conv_weight, conv_bias

    def forward(self, x: torch.Tensor, weights: torch.Tensor, use_argmax=False) -> torch.Tensor:
        weight, bias = self.compute_weight_and_bias_mixture(
            weights, self.layer.weight, self.layer.bias, use_argmax=use_argmax)
        x = F.layer_norm(x, (x.shape[-1], ),
                         weight=weight,
                         bias=bias,
                         eps=self.layer.eps)

        return x


class GPSampled(torch.nn.Module):

    def __init__(self, emb_choice: int, super_emb: int) -> None:
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x[:, :, :self.emb_choice][:, 1:], dim=1)
        if x.shape[-1] != self.super_emb:
            x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)
        return x


class LinearSampled(torch.nn.Module):

    def __init__(self, layer: torch.nn.Linear, emb_choice: int,
                 super_emb: int) -> None:
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.layer.weight[:, :self.emb_choice]
        bias = self.layer.bias
        x = F.linear(x[:, :self.emb_choice], weight=weight, bias=bias)
        return x


class LinearMixture(torch.nn.Module):

    def __init__(self, layer: torch.nn.Linear, emb_choice_list: list) -> None:
        super().__init__()
        self.emb_choice_list = emb_choice_list
        self.max_emb = max(emb_choice_list)
        self.layer = layer

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, op_weight, op_bias, use_argmax=False):
        alpha = weights[idx]
        emb_size = self.emb_choice_list[idx]
        start = 0
        end = start + emb_size
        # print(op_weight.shape)
        weight_curr = alpha * op_weight[:, :end]
        if not use_argmax:
            conv_weight += F.pad(weight_curr, (0, self.max_emb-emb_size, 0, 0), "constant", 0)
        else:
            conv_weight += weight_curr

        if op_bias is not None:
            bias = op_bias
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
        if op_bias == None:
            conv_bias = op_bias
        return conv_weight, conv_bias

    def forward(self, x: torch.Tensor, weights: torch.Tensor, use_argmax=False) -> torch.Tensor:
        # print(x.shape)
        # print(self.layer.weight.shape)
        # print(self.layer.bias.shape)
        weight, bias = self.compute_weight_and_bias_mixture(
            weights, self.layer.weight, self.layer.bias, use_argmax=use_argmax)
        x = F.linear(x, weight=weight, bias=bias)
        return x


class DropoutSampled(torch.nn.Module):

    def __init__(self, emb_choice: int, super_emb: int,
                 super_dropout: float) -> None:
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb
        self.super_dropout = super_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_dropout = calc_dropout(self.super_dropout, self.emb_choice,
                                      self.super_emb)
        x = F.dropout(x[:, :, :self.emb_choice],
                      p=sample_dropout,
                      training=self.training)
        if x.shape[-1] != self.super_emb:
            x = F.pad(x, (0, self.super_emb - x.shape[-1]), "constant", 0)
        return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class EmbedMask(nn.Module):

    def __init__(self, emb_choice: int, super_emb: int) -> None:
        super().__init__()
        self.emb_choice = emb_choice
        self.super_emb = super_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros([self.super_emb], device=x.device)
        mask[:self.emb_choice] = 1
        # print(x.shape)
        return x * mask


class Vision_TransformerSuper(nn.Module):

    def __init__(self,
                 optimizer: str = "darts",
                 config: dict = {},
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 240,
                 depth: int = 14,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: int = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 pre_norm: bool = True,
                 scale: bool = False,
                 gp: bool = False,
                 relative_position: bool = False,
                 change_qkv: bool = False,
                 abs_pos: bool = True,
                 max_relative_position: int = 14,
                 use_we_v2: bool = False) -> None:
        super(Vision_TransformerSuper, self).__init__()
        # the configs of super arch
        # self.super_embed_dim = args.embed_dim
        self.config = config
        self._initialize_alphas()
        self.mixop = get_mixop(optimizer, use_we_v2=use_we_v2)
        self.sampler = get_sampler(optimizer)
        self.img_size = img_size
        self.use_we_v2 = use_we_v2
        self.change_qkv = change_qkv
        self.max_relative_position = max_relative_position
        self.super_embed_dim = max(self.config["embed_dim"])
        self.super_mlp_ratio = max(self.config["mlp_ratio"])
        self.super_layer_num = max(self.config["layer_num"])
        self.super_num_heads = max(self.config["num_heads"])
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.pre_norm = pre_norm
        self.scale = scale
        self.qkv_bias = qkv_bias
        self.optimizer = optimizer
        patch_embed_super = PatchembedSuper(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.super_embed_dim,
            scale=scale,
            abs_pos=abs_pos,
            super_dropout=self.super_dropout,
            pre_norm=pre_norm)
        if self.use_we_v2:
            self.op_emb = PatchembedMixture(
                patch_embed_super, emb_choice_list=self.config["embed_dim"])
            self.patch_emb_op_list = self._init_entangled_op(
                self.op_emb, choices=self.config["embed_dim"], op_name="patch_embed")
        else:
            self.patch_emb_op_list = torch.nn.ModuleList([
                PatchembedSub(patch_embed_super, e)
                for e in self.config["embed_dim"]])

        self.relative_position = relative_position
        self.gp = gp
        # self.optimizer = optimizer
        self.blocks = nn.ModuleList()
        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, self.super_layer_num)
        ]  # stochastic depth decay rule
        for i in range(self.super_layer_num):
            self.blocks.append(
                TransformerEncoderLayer(
                    self.mixop,
                    config,
                    dim=self.super_embed_dim,
                    num_heads=self.super_num_heads,
                    mlp_ratio=self.super_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    scale=self.scale,
                    change_qkv=change_qkv,
                    relative_position=relative_position,
                    max_relative_position=max_relative_position,
                    use_we_v2=use_we_v2))

        # parameters for vision transformer
        num_patches = patch_embed_super.num_patches
        self.abs_pos = abs_pos
        # if self.abs_pos:
        #    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1,self.super_embed_dim))
        #    trunc_normal_(self.pos_embed, std=.02)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.super_embed_dim))
        # trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            norm = LayerNormSuper(super_embed_dim=self.super_embed_dim)
            if self.use_we_v2:
                self.op_norm = LayerNormMainMixture(norm, self.config["embed_dim"])
                self.norm_emb_op_list = self._init_entangled_op(
                    self.op_norm, self.config["embed_dim"], "layer_norm")
            else:
                self.norm_emb_op_list = [
                    LayerNormSampled(norm, e, self.super_embed_dim)
                    for e in self.config["embed_dim"]]

        # classifier head
        if not self.use_we_v2:
            self.gp_op_list = [
                GPSampled(e, self.super_embed_dim)
                for e in self.config["embed_dim"]]
        head = LinearSuper(
            self.super_embed_dim,
            num_classes) if num_classes > 0 else nn.Identity()
        if self.use_we_v2:
            self.op_head = LinearMixture(head, self.config["embed_dim"])
            self.head_op_list = self._init_entangled_op(
                self.op_head, self.config["embed_dim"], "embed_dim")
        else:
            self.head_op_list = torch.nn.ModuleList([
                LinearSampled(head, e, self.super_embed_dim)
                for e in self.config["embed_dim"]
            ])
        self.apply(self._init_weights)

    def _init_entangled_op(self, op, choices, op_name):
        ops = [EntangledOp(op=None, name=op_name)
               for i in choices[:-1]] + [EntangledOp(op=op, name=op_name)]
        return ops

    def _init_entangled_op_cross(self, op, choices1, choices2, op_name):
        choices_cross = list(itertools.product(choices1, choices2))
        ops = [EntangledOp(op=None, name=op_name)
               for i, j in choices_cross[:-1]] + [EntangledOp(op=op, name=op_name)]
        return ops

    def _initialize_alphas(self) -> None:

        self.alphas_embed_dim = torch.nn.Parameter(
        1e-3 * torch.randn(1, len(self.config["embed_dim"])),
        requires_grad=True)
        self.alphas_mlp_ratio = torch.nn.Parameter(1e-3 * torch.randn(
            max(self.config["layer_num"]), len(self.config["mlp_ratio"])),
            requires_grad=True)
        self.alphas_num_heads = torch.nn.Parameter(1e-3 * torch.randn(max(self.config["layer_num"]), len(self.config["num_heads"])),
                                                   requires_grad=True)
        self.alphas_layer_num = torch.nn.Parameter(
            1e-3 * torch.randn(1, len(self.config["layer_num"])), requires_grad=True)
        self._arch_parameters = [
            self.alphas_embed_dim, self.alphas_mlp_ratio,
            self.alphas_num_heads, self.alphas_layer_num
        ]

    def arch_parameters(self) -> list:
        return self._arch_parameters

    def _init_weights(self, m: torch.Tensor) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> dict:
        return {
            'patch_embed_super.pos_embed', 'patch_embed_super.cls_token',
            'rel_pos_embed'
        }

    def get_classifier(self) -> torch.nn.Module:
        return self.head

    def reset_classifier(self,
                         num_classes: int,
                         global_pool: str = '') -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict) -> None:
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout,
                                           self.sample_embed_dim[0],
                                           self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [
            out_dim for out_dim in self.sample_embed_dim[1:]
        ] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout,
                                              self.sample_embed_dim[i],
                                              self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout,
                                                   self.sample_embed_dim[i],
                                                   self.super_embed_dim)
                blocks.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[i],
                    sample_mlp_ratio=self.sample_mlp_ratio[i],
                    sample_num_heads=self.sample_num_heads[i],
                    sample_dropout=sample_dropout,
                    sample_out_dim=self.sample_output_dim[i],
                    sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1],
                                    self.num_classes)

    def get_sampled_params_numel(self, config: dict) -> float:
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(
                        name.split('.')[1]) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0] * (
            2 + self.patch_embed_super.num_patches)

    def get_complexity(self, sequence_length: int) -> float:
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(
            self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops

    def forward_features(self, x: torch.Tensor,
                         weights_embed_dim: torch.Tensor,
                         weights_mlp_ratio: torch.Tensor,
                         weights_num_heads: torch.Tensor,
                         weight_layer_num: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.mixop.forward(x, weights_embed_dim[0], self.patch_emb_op_list)
        i = 0
        x_list = []
        for blk in self.blocks:
            x = blk(x, i, weights_embed_dim, weights_mlp_ratio,
                    weights_num_heads)
            i = i + 1
            print(x)
            if i in self.config["layer_num"]:
                x_list.append(x)
            # print(torch.cuda.memory_allocated("cuda"))
        x = self.mixop.forward_depth(x_list, weight_layer_num[0])
        # print(time.time()-start_time)
        if self.pre_norm:
            # print(x.shape)
            x = self.mixop.forward(x, weights_embed_dim[0],
                                   self.norm_emb_op_list)

        if self.gp:
            if not self.use_we_v2:
                return self.mixop.forward(
                    x, weights_embed_dim[0],
                    self.gp_op_list)  # TODO check what happens here?
            else:
                x = torch.mean(x[:, :, :][:, 1:], dim=1)
                return x

        return x[:, 0]
    
    def get_arch_parameters(self) -> list:
        return self._arch_parameters
    
    def get_network_parameters(self) -> list:
        network_params = set(self.parameters())-set(self._arch_parameters)
        return list(network_params)

    def get_named_network_parameters(self) -> dict:
        named_network_params = {}
        for name, param in self.named_parameters():
            if "arch" not in name:
                named_network_params[name] = param
        return named_network_params
    
    def forward(self,
                x: torch.Tensor,
                tau_curr: torch.Tensor,
                arch_params_sampled: list = None) -> torch.Tensor:
        if arch_params_sampled == None:
            arch_params_sampled = self.sampler.sample_step(
                self._arch_parameters)
        weights_embed_dim = arch_params_sampled[0]
        weights_mlp_ratio = arch_params_sampled[1]
        weights_num_heads = arch_params_sampled[2]
        weights_layer_num = arch_params_sampled[3]
        x = self.forward_features(x, weights_embed_dim, weights_mlp_ratio,
                                  weights_num_heads, weights_layer_num)
        x = self.mixop.forward(x, weights_embed_dim[0], self.head_op_list)
        return x

    def get_best_config(self) -> dict:
        best_config = {}
        best_config["embed_dim"] = self.config["embed_dim"][torch.argmax(
            self.alphas_embed_dim, dim=-1)]
        best_config["layer_num"] = self.config["layer_num"][torch.argmax(
            self.alphas_layer_num, dim=-1)]
        for d in range(max(self.config["layer_num"])):
            best_config["num_heads_" +
                        str(d)] = self.config["num_heads"][torch.argmax(
                            self.alphas_num_heads[d], dim=-1)]
            best_config["mlp_ratio_" +
                        str(d)] = self.config["mlp_ratio"][torch.argmax(
                            self.alphas_mlp_ratio[d], dim=-1)]
        return best_config
                

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self,
                 mixop: MixOp,
                 config: dict,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 dropout: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: torch.nn.Module = nn.GELU,
                 pre_norm: bool = True,
                 scale: bool = False,
                 relative_position: bool = False,
                 change_qkv: bool = False,
                 max_relative_position: int = 14,
                 use_we_v2: bool = False) -> None:
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.use_we_v2 = use_we_v2
        self.mixop = mixop
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)
        self.config = config
        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(mixop,
                                   config,
                                   dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=dropout,
                                   scale=self.scale,
                                   relative_position=self.relative_position,
                                   change_qkv=change_qkv,
                                   max_relative_position=max_relative_position,
                                   use_we_v2=use_we_v2)
        if not self.use_we_v2:
            self.embed_mask = torch.nn.ModuleList([
                EmbedMask(e, self.super_embed_dim)
                for e in self.config["embed_dim"]])
        attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        if self.use_we_v2:
            self.op_attn_layer_norm = LayerNormMixture(
                attn_layer_norm, self.config["embed_dim"], pre_norm, before=True)
            self.attn_layer_norm_list_before = self._init_entangled_op(
                self.op_attn_layer_norm, self.config["embed_dim"], "embed_dim")
        else:
            self.attn_layer_norm_list_before = torch.nn.ModuleList([
                LayerNormSub(attn_layer_norm,
                             e,
                             self.super_embed_dim,
                             pre_norm,
                             before=True) for e in self.config["embed_dim"]])
        if self.use_we_v2:
            self.op_ln = LayerNormMixture(
                attn_layer_norm, self.config["embed_dim"], pre_norm, after=True)
            self.attn_layer_norm_list_after = self._init_entangled_op(
                self.op_ln, self.config["embed_dim"], "embed_dim")
        else:
            self.attn_layer_norm_list_after = torch.nn.ModuleList([
                LayerNormSub(attn_layer_norm,
                             e,
                             self.super_embed_dim,
                             pre_norm,
                             after=True) for e in self.config["embed_dim"]])
        ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        if self.use_we_v2:
            self.op_ffn_ln = LayerNormMixture(
                ffn_layer_norm, self.config["embed_dim"], pre_norm, before=True)
            self.ffn_layer_norm_list_before = self._init_entangled_op(
                self.op_ffn_ln, self.config["embed_dim"], "embed_dim")
        else:
            self.ffn_layer_norm_list_before = torch.nn.ModuleList([
                LayerNormSub(ffn_layer_norm,
                             e,
                             self.super_embed_dim,
                             pre_norm,
                             before=True) for e in self.config["embed_dim"]])
        if self.use_we_v2:
            self.op_ffn_ln2 = LayerNormMixture(
                ffn_layer_norm, self.config["embed_dim"], pre_norm, after=True)
            self.ffn_layer_norm_list_after = self._init_entangled_op(
                self.op_ffn_ln2, self.config["embed_dim"], "embed_dim")
        else:
            self.ffn_layer_norm_list_after = torch.nn.ModuleList([
                LayerNormSub(ffn_layer_norm,
                             e,
                             self.super_embed_dim,
                             pre_norm,
                             after=True) for e in self.config["embed_dim"]])
        if not self.use_we_v2:
            self.dropout_op_list = [
                DropoutSampled(e, self.super_embed_dim, dropout)
                for e in self.config["embed_dim"]]

        fc1 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer)
        if self.use_we_v2:
            self.op_linear_mix = LinearSubRatioMixture(
                fc1, self.config["embed_dim"], self.config["mlp_ratio"])
            self.emb_ratio_op_fc1_list = self._init_entangled_op_cross(
                self.op_linear_mix, self.config["embed_dim"], self.config["mlp_ratio"], "embed_mlpratio")
        else:
            self.emb_ratio_op_fc1_list = torch.nn.ModuleList()
            for e in self.config["embed_dim"]:
                for r in self.config["mlp_ratio"]:
                    self.emb_ratio_op_fc1_list.append(
                        LinearSubRatioEmb(
                            fc1, self.super_embed_dim,
                            self.super_embed_dim * self.super_mlp_ratio, e, r))

        if not self.use_we_v2:
            self.dropout_op_list_fc1 = torch.nn.ModuleList()
            for e in self.config["embed_dim"]:
                for r in self.config["mlp_ratio"]:
                    self.dropout_op_list_fc1.append(
                        DropoutSampled(int(e * r),
                                       self.super_ffn_embed_dim_this_layer,
                                       dropout))

        fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=self.super_embed_dim)
        if self.use_we_v2:
            self.op_linear_mix_2 = LinearSubRatioMixture(
                fc2, self.config["embed_dim"], self.config["mlp_ratio"], reverse=True)
            self.emb_ratio_op_fc2_list = self._init_entangled_op_cross(
                self.op_linear_mix_2, self.config["embed_dim"], self.config["mlp_ratio"], "embed_mlpratio")
        else:
            self.emb_ratio_op_fc2_list = torch.nn.ModuleList()
            for e in self.config["embed_dim"]:
                for r in self.config["mlp_ratio"]:
                    self.emb_ratio_op_fc2_list.append(
                        LinearSubRatioEmb(fc2,
                                          self.super_embed_dim,
                                          self.super_embed_dim *
                                          self.super_mlp_ratio,
                                          e,
                                          r,
                                          reverse=True))
        self.activation_fn = gelu
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, id: int,
                weights_embed_dim: torch.Tensor,
                weights_mlp_ratio: torch.Tensor,
                weights_num_heads: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x
        residual = x
        x = self.mixop.forward(x, weights_embed_dim[0],
                               self.attn_layer_norm_list_before)
        x = self.attn(x, id, weights_embed_dim, weights_num_heads)
        if self.use_we_v2:
            x = F.dropout(x, p=self.super_dropout, training=self.training)
        else:
            x = self.mixop.forward(
                x, weights_embed_dim[0], self.dropout_op_list)
        x = self.drop_path(x)
        x = residual + x
        x = self.mixop.forward(x, weights_embed_dim[0],
                               self.attn_layer_norm_list_after)
        residual = x
        x = self.mixop.forward(x, weights_embed_dim[0],
                               self.ffn_layer_norm_list_before)
        weights = [weights_embed_dim[0], weights_mlp_ratio[id]]
        x = self.mixop.forward(x,
                               weights,
                               self.emb_ratio_op_fc1_list,
                               combi=True)
        x = self.activation_fn(x)
        if self.use_we_v2:
            x = F.dropout(x, p=self.super_dropout, training=self.training)
        else:
            x = self.mixop.forward(x,
                                   weights,
                                   self.dropout_op_list_fc1,
                                   combi=True)
        x = self.mixop.forward(x,
                               weights,
                               self.emb_ratio_op_fc2_list,
                               combi=True)
        if self.use_we_v2:
            x = F.dropout(x, p=self.super_dropout, training=self.training)
        else:
            x = self.mixop.forward(
                x, weights_embed_dim[0], self.dropout_op_list)
        if self.scale and not self.use_we_v2:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.mixop.forward(x, weights_embed_dim[0],
                               self.ffn_layer_norm_list_after)
        return x

    def _init_entangled_op(self, op, choices, op_name):
        ops = [EntangledOp(op=None, name=op_name)
               for i in choices[:-1]] + [EntangledOp(op=op, name=op_name)]
        return ops

    def _init_entangled_op_cross(self, op, choices1, choices2, op_name):
        choices_cross = list(itertools.product(choices1, choices2))
        ops = [EntangledOp(op=None, name=op_name)
               for i, j in choices_cross[:-1]] + [EntangledOp(op=op, name=op_name)]
        return ops

    def maybe_layer_norm(self,
                         layer_norm: torch.nn.LayerNorm,
                         x: torch.Tensor,
                         before: bool = False,
                         after: bool = False) -> torch.Tensor:
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def get_complexity(self, sequence_length: int) -> float:
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc1.get_complexity(sequence_length + 1)
        total_flops += self.fc2.get_complexity(sequence_length + 1)
        return total_flops
    



def calc_dropout(dropout: float, sample_embed_dim: int,
                 super_embed_dim: int) -> float:
    return dropout * 1.0 * sample_embed_dim / super_embed_dim
