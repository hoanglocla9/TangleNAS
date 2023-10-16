# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
import copy
import random
import numpy as np
import torch.nn.functional as F

from search_spaces.MobileNetV3.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
)
from search_spaces.MobileNetV3.modules.dynamic_layers import (
    DynamicMBConvLayer, DynamicMBConvLayerMixture
)
from torch.autograd import Variable
from search_spaces.MobileNetV3.mobilenetv3 import MobileNetV3
from search_spaces.MobileNetV3.utils import make_divisible, val2list, MyNetwork
from optimizers.optim_factory import get_mixop, get_sampler

__all__ = ["OFAMobileNetV3"]


class OFAMobileNetV3Mixture(MobileNetV3):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        base_stage_width=None,
        width_mult=1.0,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[1, 2, 3],
        optimizer = "darts_v1"
    ):

        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        self.mixop = get_mixop(optimizer,use_we_v2=True)
        self.sampler = get_sampler(optimizer)

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = make_divisible(
            base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        last_channel = make_divisible(
            base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []

        for base_width in base_stage_width[:-2]:
            width = make_divisible(
                base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
            )
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, act_func="h_swish"
        )
        first_block_conv = MBConvLayer(
            in_channels=input_channel,
            out_channels=first_block_dim,
            kernel_size=3,
            stride=stride_stages[0],
            expand_ratio=1,
            act_func=act_stages[0],
            use_se=se_stages[0],
        )
        first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(first_block_dim, first_block_dim)
            if input_channel == first_block_dim
            else None,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = first_block_dim
        
        for width, n_block, s, act_func, use_se in zip(
            width_list[2:],
            n_block_list[1:],
            stride_stages[1:],
            act_stages[1:],
            se_stages[1:],
        ):
            self.block_group_info.append(
                [_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayerMixture(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    mixop=self.mixop,
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        # final expand layer, feature mix layer & classifier
        final_expand_layer = ConvLayer(
            feature_dim, final_expand_width, kernel_size=1, act_func="h_swish"
        )
        feature_mix_layer = ConvLayer(
            final_expand_width,
            last_channel,
            kernel_size=1,
            bias=False,
            use_bn=False,
            act_func="h_swish",
        )

        classifier = LinearLayer(
            last_channel, n_classes, dropout_rate=dropout_rate)

        super(OFAMobileNetV3Mixture, self).__init__(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        self._init_arch_parameters()
        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx)
                              for block_idx in self.block_group_info]
        
        self.model_name = "mobilenetv3_default"

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return "OFAMobileNetV3ixture"

    def get_depth_mixture(self, all_features, depth_weights):
        out = 0
        for i, w in enumerate(depth_weights):
            out = out + all_features[i] * w
        return out

    def _init_arch_parameters(self):
        self.arch_kernel_size_weights = torch.nn.Parameter(1e-3 * torch.randn(
            [len(self.block_group_info), max(self.depth_list), len(self.ks_list)]),
            requires_grad=True
        )
        self.arch_expand_ratio_weights = torch.nn.Parameter(1e-3 * torch.randn(
            [
                len(self.block_group_info),
                max(self.depth_list),
                len(self.expand_ratio_list)
            ]),
            requires_grad=True
        )
        self.arch_depth_weights = torch.nn.Parameter(1e-3 * torch.randn(
            [len(self.block_group_info), len(self.depth_list)]),
            requires_grad=True
        )
        #self.register_buffer('arch_kernel_size_weights', arch_kernel_size_weights)
        #self.register_buffer('arch_expand_ratio_weights', arch_expand_ratio_weights)
        #self.register_buffer('arch_depth_weights', arch_depth_weights)

    def state_dict_archs(self):
        return {'arch_kernel_size_weights': self.arch_kernel_size_weights, 'arch_expand_ratio_weights': self.arch_expand_ratio_weights, 'arch_depth_weights': self.arch_depth_weights}
    
    def arch_parameters(self):
        return [self.arch_depth_weights, self.arch_expand_ratio_weights, self.arch_kernel_size_weights]
    
    def model_weights(self):
        model_weights = set(model_weights.parameters())-set(self.arch_parameters())
        return model_weights
    
    def get_model_params(self, key, include=False):
        param_list = []
        if include:
            for name, param in self.named_parameters():
                if (key in name) and ('arch' not in name):
                    param_list.append(param)
        else:
            for name, param in self.named_parameters():
                if (key not in name) and ('arch' not in name):
                    param_list.append(param)
        return param_list

    def get_arch_represenation(self):
        def get_argmaxes(weights):
            # count the number of dimensions in weights
            num_dims = len(weights.shape)

            assert num_dims == 2 or num_dims == 3, "weights must be 2 or 3 dimensional"

            if num_dims == 2:
                return torch.argmax(weights, dim=1, keepdim=False).detach().cpu().numpy()
            elif num_dims == 3:
                arg_maxs = []

                for block_weights in weights:
                    arg_maxs.append(torch.argmax(block_weights, dim=1, keepdim=False).detach().cpu().tolist())

                return np.array(arg_maxs)

        def convert_to_genotype(alphas_argmax, operations):
            return np.array(
                [operations[x] for x in alphas_argmax.reshape(-1).tolist()]
            ).reshape(alphas_argmax.shape).tolist()

        def convert_genotype_to_dict(genotype):
            blocks = {}

            g = np.array(genotype).squeeze()
            num_dims = len(g.shape)

            if num_dims == 1:
                return {
                    "block_0": genotype
                }

            for i, block in enumerate(g):
                blocks[f"block_{i}"] = block.tolist()

            return blocks

        arch_configs = [
            (self.arch_kernel_size_weights, self.ks_list, "kernel_size"),
            (self.arch_expand_ratio_weights, self.expand_ratio_list, "expand_ratio"),
            (self.arch_depth_weights, self.depth_list, "depth"),
        ]

        genotype_draft = {}

        for arch_params, choices, op_name in arch_configs:
            arg_maxs = get_argmaxes(arch_params)
            genotype = convert_to_genotype(arg_maxs, choices)
            genotype_draft[op_name] = genotype

        representation = {}

        for k, v in genotype_draft.items():
            representation[k] = convert_genotype_to_dict(v)

        return representation

    def forward(self, x):
        # first conv
        if self.training is True:
            weights = self.sampler.sample_step(self.arch_parameters())
        else:
            weights = [F.softmax(w, dim=-1) for w in self.arch_parameters()]

        depth_weights , expand_ratio_weights, kernel_size_weights = weights[0], weights[1], weights[2]
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)
        # blocks
        for i, block_idx in enumerate(self.block_group_info):
            depths_collector = []
            count = 0
            for idx in list(block_idx):
                x = self.blocks[idx](x, expand_ratio_weights[i,count,:], kernel_size_weights[i,count,:])
                if count+1 in self.depth_list:
                    depths_collector.append(x)
                count = count+1
            x = self.get_depth_mixture(depths_collector, depth_weights[i])
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(
            2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"

        _str += self.final_expand_layer.module_str + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": OFAMobileNetV3Mixture.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "final_expand_layer": self.final_expand_layer.config,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if ".mobile_inverted_conv." in key:
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif ".bn.bn." in new_key:
                new_key = new_key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in new_key:
                new_key = new_key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in new_key:
                new_key = new_key.replace(".linear.linear.", ".linear.")
            ##############################################################################
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAMobileNetV3Mixture, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list)
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type="depth"):
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None

    def sample_active_subnet(self):
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [
                ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [
                expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]

        final_expand_layer = copy.deepcopy(self.final_expand_layer)
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    ResidualBlock(
                        self.blocks[idx].conv.get_active_subnet(
                            input_channel, preserve_weight
                        ),
                        copy.deepcopy(self.blocks[idx].shortcut),
                    )
                )
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        final_expand_config = self.final_expand_layer.config
        feature_mix_layer_config = self.feature_mix_layer.config
        classifier_config = self.classifier.config

        block_config_list = [first_block_config]
        input_channel = first_block_config["conv"]["out_channels"]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    {
                        "name": ResidualBlock.__name__,
                        "conv": self.blocks[idx].conv.get_active_subnet_config(
                            input_channel
                        ),
                        "shortcut": self.blocks[idx].shortcut.config
                        if self.blocks[idx].shortcut is not None
                        else None,
                    }
                )
                input_channel = self.blocks[idx].conv.active_out_channel
            block_config_list += stage_blocks

        return {
            "name": MobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": first_conv_config,
            "blocks": block_config_list,
            "final_expand_layer": final_expand_config,
            "feature_mix_layer": feature_mix_layer_config,
            "classifier": classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)

'''from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1235'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="drnas").to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(2,3,224,224)).to(rank)
    labels = torch.randn(2, 1000).to(rank)
    if rank==0:
        print(loss_fn(outputs, labels))
    loss_fn(outputs, labels).backward()
    for name, param in ddp_model.named_parameters():
        if param.grad==None:
            print(name, param.grad)
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 8)'''
'''model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="drnas")
torch.save(model.state_dict(), "ofa_mbv3.pth")
state_dict = torch.load("ofa_mbv3.pth")
#for k in state_dict:
#    print(k)
outputs = model(torch.randn(2,3,224,224))
optimizer = torch.optim.SGD(model.arch_parameters(), lr=0.001)
loss = torch.sum(outputs)
loss.backward()
print("Before", model.arch_depth_weights)
for p in model.arch_parameters():
    p.grad = torch.ones_like(p)*10
optimizer.step()
print("After", model.arch_depth_weights)
torch.save(model.arch_parameters(), "ofa_mbv3_arch.pth")
print(torch.load("ofa_mbv3_arch.pth"))'''
#for n,p in model.named_parameters():
#    print(n)
#for n,p in model.model_weights():
#    print(n)

'''from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1235'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="drnas").to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(2,3,224,224)).to(rank)
    labels = torch.randn(2, 1000).to(rank)
    if rank==0:
        print(loss_fn(outputs, labels))
    loss_fn(outputs, labels).backward()
    for name, param in ddp_model.named_parameters():
        if param.grad==None:
            print(name, param.grad)
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 8)'''
'''model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="drnas")
torch.save(model.state_dict(), "ofa_mbv3.pth")
state_dict = torch.load("ofa_mbv3.pth")
#for k in state_dict:
#    print(k)
outputs = model(torch.randn(2,3,224,224))
optimizer = torch.optim.SGD(model.arch_parameters(), lr=0.001)
loss = torch.sum(outputs)
loss.backward()
print("Before", model.arch_depth_weights)
for p in model.arch_parameters():
    p.grad = torch.ones_like(p)*10
optimizer.step()
print("After", model.arch_depth_weights)
torch.save(model.arch_parameters(), "ofa_mbv3_arch.pth")
print(torch.load("ofa_mbv3_arch.pth"))'''
#for n,p in model.named_parameters():
#    print(n)
#for n,p in model.model_weights():
#    print(n)