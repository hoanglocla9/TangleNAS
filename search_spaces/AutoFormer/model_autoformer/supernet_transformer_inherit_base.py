import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_autoformer.module.Linear_super import LinearSuper
from model_autoformer.module.layernorm_super import LayerNormSuper
from model_autoformer.module.multihead_super import AttentionSuper
from model_autoformer.module.embedding_super import PatchembedSuper
from model_autoformer.utils import trunc_normal_
from model_autoformer.utils import DropPath
import numpy as np


def derive_best_config(config):
    config_best = {}
    config_best["layer_num"] = int(config["layer_num"])
    config_best["embed_dim"] = [int(config["embed_dim"])] * int(
        config["layer_num"])
    config_best["mlp_ratio"] = []  #config["layer_num"]
    config_best["num_heads"] = []
    for i in range(int(config_best["layer_num"])):
        config_best["num_heads"].append(int(config["num_heads_" + str(i)]))
        config_best["mlp_ratio"].append(int(config["mlp_ratio_" + str(i)]))
    return config_best


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_TransformerSuper(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=240,
                 depth=14,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pre_norm=True,
                 scale=False,
                 gp=True,
                 relative_position=True,
                 change_qkv=False,
                 abs_pos=True,
                 max_relative_position=14):
        super(Vision_TransformerSuper, self).__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.scale = scale
        self.patch_embed_super = PatchembedSuper(img_size=img_size,
                                                 patch_size=patch_size,
                                                 in_chans=in_chans,
                                                 embed_dim=embed_dim)
        self.gp = gp

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(
                TransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    scale=self.scale,
                    change_qkv=change_qkv,
                    relative_position=relative_position,
                    max_relative_position=max_relative_position))

        # parameters for vision transformer
        num_patches = self.patch_embed_super.num_patches

        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classifier head
        self.head = LinearSuper(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

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
            max(self.config["layer_num"]), len(self.config["mlp_ratio"])),
                                                   requires_grad=True)
        self.alphas_num_heads = torch.nn.Parameter(1e-3 * torch.randn(
            max(self.config["layer_num"]), len(self.config["num_heads"])),
                                                   requires_grad=True)
        self.alphas_layer_num = torch.nn.Parameter(
            1e-3 * torch.randn(1, len(self.config["layer_num"])),
            requires_grad=True)
        self._arch_parameters = [
            self.alphas_embed_dim, self.alphas_mlp_ratio,
            self.alphas_num_heads, self.alphas_layer_num
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def print_alphas(self):
        print("Layer num alphas",
              torch.nn.functional.softmax(self.alphas_layer_num, dim=-1))
        print("Head num alphas",
              torch.nn.functional.softmax(self.alphas_num_heads, dim=-1))
        print("Mlp ratio num alphas",
              torch.nn.functional.softmax(self.alphas_mlp_ratio, dim=-1))
        print("Embed dim alphas",
              torch.nn.functional.softmax(self.alphas_embed_dim, dim=-1))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def sample(self, use_alphas=True):
        alphas = {}
        alphas['mlp_ratio'] = self.alphas_mlp_ratio
        alphas['num_heads'] = self.alphas_num_heads
        alphas['layer_num'] = self.alphas_layer_num
        alphas['embed_dim'] = self.alphas_embed_dim

        sampled_config = {}

        def _sample(choices, p):
            if use_alphas:
                p = F.softmax(p.detach().squeeze(), dim=-1).cpu().numpy()
                return np.random.choice(choices, p=p)
            else:
                return np.random.choice(choices)

        depth = _sample(self.config['layer_num'], p=alphas['layer_num'])
        sampled_config['layer_num'] = depth

        dimensions = ['mlp_ratio', 'num_heads']

        for dimension in dimensions:
            sampled_config[dimension] = [
                _sample(self.config[dimension], p=alphas[dimension][idx])
                for idx in range(depth)
            ]

        sampled_config['embed_dim'] = [
            _sample(self.config['embed_dim'], p=alphas['embed_dim'])
        ] * depth

        return sampled_config

    def set_sample_config(self, config: dict):
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

    def get_sampled_params_numel(self, config):
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

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(
            self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed_super(x)

        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(
            B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., :self.sample_embed_dim[0]]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        #print(torch.cuda.memory_allocated("cuda"))
        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
            #print(torch.cuda.memory_allocated("cuda"))
        # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x)

        if self.gp:
            return torch.mean(x[:, 1:], dim=1)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 pre_norm=True,
                 scale=False,
                 relative_position=False,
                 change_qkv=False,
                 max_relative_position=14):
        super().__init__()
        #change_qkv = True
        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.is_identity_layer = None
        #print(change_qkv)
        self.attn = AttentionSuper(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=dropout,
                                   scale=self.scale,
                                   relative_position=self.relative_position,
                                   change_qkv=change_qkv,
                                   max_relative_position=max_relative_position)

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=self.super_embed_dim)

    def set_sample_config(self,
                          is_identity_layer,
                          sample_embed_dim=None,
                          sample_mlp_ratio=None,
                          sample_num_heads=None,
                          sample_dropout=None,
                          sample_attn_dropout=None,
                          sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim *
                                                   sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(
            sample_q_embed_dim=self.sample_num_heads_this_layer * 64,
            sample_num_heads=self.sample_num_heads_this_layer,
            sample_in_embed_dim=self.sample_embed_dim)

        self.fc1.set_sample_config(
            sample_in_dim=self.sample_embed_dim,
            sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer,
            sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)
        # compute the ffn
        # start_time = time.time()
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc1.get_complexity(sequence_length + 1)
        total_flops += self.fc2.get_complexity(sequence_length + 1)
        return total_flops


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


def count_parameters_in_MB(model):
    return np.sum(
        np.prod(v.size()) for name, v in model.named_parameters()
        if "auxiliary" not in name) / 1e6


