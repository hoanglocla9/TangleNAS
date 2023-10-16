#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
from typing import Text, Any
import torch.nn as nn
import torch
from search_spaces.NATS.operations import ResNetBasicblock, ResNetBasicblockSub
from search_spaces.NATS.cells import InferCellDiscretize
import torch.nn.functional as F
from optimizers.optim_factory import get_mixop, get_sampler
from torch.autograd import Variable
import torch
from torch.distributions.dirichlet import Dirichlet
import itertools
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
from search_spaces.NATS.genotypes import Structure as CellStructure
from search_spaces.base_model import NetworkBase


class StemSampled(torch.nn.Module):

    def __init__(self, C, C_max):
        super().__init__()
        self.C = C
        self.bn_sampled = BatchNormSampled(C, C_max)

    def forward(self, x, stem):
        x = torch.nn.functional.conv2d(x,
                                       stem[0].weight[:self.C, :, :, :],
                                       bias=stem[0].bias,
                                       stride=stem[0].stride,
                                       padding=stem[0].padding)
        x = self.bn_sampled(x, stem[1])
        return x


class LinearSampled(torch.nn.Module):

    def __init__(self, C_in, C_max):
        super().__init__()
        self.C_in = C_in
        self.C_max = C_max

    def forward(self, x, layer):
        weight = layer.weight[:, :self.C_in]
        bias = layer.bias
        #print(x.shape)
        #print(weight.shape)
        x = F.linear(x[:, :self.C_in], weight=weight, bias=bias)
        return x


class AdaptivePoolingSampled(torch.nn.Module):

    def __init__(self, C, C_max):
        super().__init__()
        self.C_max = C_max
        self.C = C

    def forward(self, x, layer):
        x = layer(x[:, :self.C, :, :])
        x = x.view(x.size(0), -1)
        #print(x.shape)

        x = F.pad(x, (0, self.C_max - self.C, 0, 0), "constant", 0)

        return x


class ConvSampledOut(torch.nn.Module):

    def __init__(self, channel, max_channels):
        super().__init__()
        self.channel_choice = channel
        self.max_channels = max_channels

    def forward(self, x):
        mask = torch.zeros([1, self.max_channels, 1, 1], device=x.device)
        mask[:, :self.channel_choice, :, :] = 1
        return x * mask


class BatchNormSampled(torch.nn.Module):

    def __init__(self, channel, max_channels):
        super().__init__()
        self.channel_choice = channel
        self.max_channels = max_channels

    def forward(self, x, bn):
        if bn.training:
            bn_training = True
        else:
            bn_training = (bn.running_mean is None) and (bn.running_var is
                                                         None)
        if bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = bn.momentum

        if bn.training and bn.track_running_stats:
            assert bn.num_batches_tracked is not None
            bn.num_batches_tracked.add_(1)
            if bn.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / bn.num_batches_tracked.item(
                )
            else:  # use exponential moving average
                exponential_average_factor = bn.momentum
        x = F.batch_norm(
            x[:, :self.channel_choice, :, :],
            # If buffers are not to be tracked,ensure that they won't be updated
            bn.running_mean[:self.channel_choice]
            if not bn.training or bn.track_running_stats else None,
            bn.running_var[:self.channel_choice]
            if not bn.training or bn.track_running_stats else None,
            bn.weight[:self.channel_choice],
            bn.bias[:self.channel_choice],
            bn_training,
            exponential_average_factor,
            bn.eps,
        )
        if self.channel_choice != self.max_channels:
            x = F.pad(x, (0, 0, 0, 0, 0, self.max_channels - x.shape[1], 0, 0),
                      "constant", 0)
        return x


class NATSModel(NetworkBase):

    def __init__(self, genotype: Any, num_classes: int, affine=True, track_running_stats=True):
        super(NATSModel, self).__init__()
        channels = [64, 64, 64, 64, 64]
        self._channels = channels
        self.genotype_best = genotype
        self.channels_choice = [8, 16, 24, 32, 40, 48, 56, 64]
        if len(channels) % 3 != 2:
            raise ValueError("invalid number of layers : {:}".format(
                len(channels)))
        self._num_stage = N = len(channels) // 3
        self.num_classes = num_classes
        self.stem = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64))
        self.stem_ops = [StemSampled(c, 64) for c in self.channels_choice]
        self.mixop = get_mixop("discrete")
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N
        self.cells = nn.ModuleList()
        self.cell_ops = []
        for _, (_, reduction) in enumerate(zip(channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(64, 64, 2, True)
                cell_ops = []
                for c_in in self.channels_choice:
                    for c_out in self.channels_choice:
                        cell_ops.append(ResNetBasicblockSub(c_in, c_out, 64))
                self.cell_ops.append(cell_ops)
            else:
                cell = InferCellDiscretize(genotype, 64, 64, 1, affine=affine, track_running_stats=track_running_stats)
            self.cells.append(cell)
        self._num_layer = len(self.cells)

        self.lastactbn = nn.BatchNorm2d(64)
        self.lastactbn_ops = [
            BatchNormSampled(c, 64) for c in self.channels_choice
        ]
        self.lastactact = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling_ops = [
            AdaptivePoolingSampled(c, 64) for c in self.channels_choice
        ]
        self.classifier = nn.Linear(64, num_classes)
        self.classifier_ops = [
            LinearSampled(c, 64) for c in self.channels_choice
        ]
        self.relu = torch.nn.ReLU()
        self._initialize_alphas()

    def before_epoch(self, drop_prob):
        pass

    def _initialize_alphas(self):
        self.alphas_level_1 = torch.nn.Parameter(1e-3 * torch.randn([8]),
                                                 requires_grad=True)
        self.alphas_level_2 = torch.nn.Parameter(1e-3 * torch.randn([8]),
                                                 requires_grad=True)
        self.alphas_level_3 = torch.nn.Parameter(1e-3 * torch.randn([8]),
                                                 requires_grad=True)
        self.alphas_level_4 = torch.nn.Parameter(1e-3 * torch.randn([8]),
                                                 requires_grad=True)
        self.alphas_level_5 = torch.nn.Parameter(1e-3 * torch.randn([8]),
                                                 requires_grad=True)
        self._arch_parameters = [
            self.alphas_level_1, self.alphas_level_2, self.alphas_level_3,
            self.alphas_level_4, self.alphas_level_5
        ]

    def forward(self, inputs, arch_params=None):
        arch_params = self.arch_parameters()
        arch_params = [arch_params[0]] + self.arch_parameters()
        feature = self.mixop.forward_layer(inputs, arch_params[0],
                                           self.stem_ops, self.stem)
        i_res = 0
        arch_param_id = 1
        #print(self.cell_ops)
        for i, cell in enumerate(self.cells):
            if cell.__class__.__name__ == "ResNetBasicblock":
                op_list = self.cell_ops[i_res]
                #print(feature.shape)
                feature = self.mixop.forward_layer(feature, [
                    arch_params[arch_param_id - 1], arch_params[arch_param_id]
                ],
                                                   op_list,
                                                   cell,
                                                   combi=True)
                i_res = i_res + 1
            else:
                feature = cell(
                    feature, arch_params[arch_param_id - 1].to(inputs.device),
                    arch_params[arch_param_id].to(inputs.device))
            arch_param_id += 1
        feature = self.mixop.forward_layer(feature,
                                           arch_params[-1].to(inputs.device),
                                           self.lastactbn_ops, self.lastactbn)
        out = self.lastactact(feature)
        out = self.mixop.forward_layer(feature,
                                       arch_params[-1].to(inputs.device),
                                       self.global_pooling_ops,
                                       self.global_pooling)
        logits = self.mixop.forward_layer(out,
                                          arch_params[-1].to(inputs.device),
                                          self.classifier_ops, self.classifier)
        return out, None, logits


#test
#model = NATSModel(genotype=CellStructure.str2structure('|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'), num_classes=10)#.cuda()
#model.load_state_dict(torch.load("/work/dlclarge1/sukthank-transformer_search/GraViT-E/model_nats.path"))
#img = torch.randn([64,3,32,32])#.cuda()
#print(model(img)[-1].shape)
