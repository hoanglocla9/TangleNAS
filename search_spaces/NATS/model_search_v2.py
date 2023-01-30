#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
from typing import List, Text, Any
import torch.nn as nn
import torch
from search_spaces.NATS.operations import ResNetBasicblock
from search_spaces.NATS.operations import ResNetBasicblockSub
from search_spaces.NATS.cells import InferCellV2
import torch.nn.functional as F
from nats_bench import create
from optimizers.optim_factory import get_mixop, get_sampler
from torch.autograd import Variable
import torch
from torch.distributions.dirichlet import Dirichlet
import itertools
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
from search_spaces.base_model_search import SearchNetworkBase
from search_spaces.NATS.genotypes import Structure as CellStructure


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


class NATSSearchSpaceV2(SearchNetworkBase):

    def __init__(self,
                 optimizer_type,
                 genotype: Any,
                 num_classes: int,
                 criterion: Any,
                 reg_type='l2',
                 reg_scale=1e-3,
                 affine=True,
                 track_running_stats=True,
                 path_to_benchmark = "",
                 initialize_api = True):
        super(NATSSearchSpaceV2, self).__init__()
        channels = [64, 64, 64, 64, 64]
        self._channels = channels
        self.genotype_best = genotype
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        self.affine = affine
        self.track_running_stats = track_running_stats
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
        self.optimizer_type = optimizer_type
        self.path_to_benchmark=path_to_benchmark
        self.sampler = get_sampler(self.optimizer_type)
        self.mixop = get_mixop(self.optimizer_type)
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N
        if initialize_api:
            self.api = create(
            path_to_benchmark,
            'sss',
            fast_mode=True,
            verbose=True)
        c_prev = channels[0]
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
                cell = InferCellV2(self.mixop, genotype, 64, 64, 1, affine=affine, track_running_stats=track_running_stats)
            self.cells.append(cell)
            c_prev = cell.out_dim
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
        self.tau = torch.Tensor([10])
        self._criterion = criterion
        self._initialize_alphas()
        self._initialize_anchors()

    def _initialize_anchors(self):
        self.anchor_1 = Dirichlet(torch.ones_like(self.alphas_level_1))
        self.anchor_2 = Dirichlet(torch.ones_like(self.alphas_level_2))
        self.anchor_3 = Dirichlet(torch.ones_like(self.alphas_level_3))
        self.anchor_4 = Dirichlet(torch.ones_like(self.alphas_level_4))
        self.anchor_5 = Dirichlet(torch.ones_like(self.alphas_level_5))

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(i, len(self.cells),
                                                       cell.extra_repr())
        return string

    def new(self):
        model_new = NATSSearchSpaceV2(self.optimizer_type, self.genotype_best,
                                      self.num_classes, self._criterion,
                                      self.reg_type, self.reg_scale,
                                      path_to_benchmark=self.path_to_benchmark,
                                      affine=self.affine, track_running_stats=self.track_running_stats,
                                      initialize_api=False).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def extra_repr(self):
        return "{name}(C={_channels}, N={_num_stage}, L={_num_layer})".format(
            name=self.__class__.__name__, **self.__dict__)

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

    def genotype(self):
        genotypes = []
        arch_params = self.arch_parameters()
        for x in arch_params:
            genotypes.append(self.channels_choice[x.argmax().item()])
        genotype_string = ""
        for c in genotypes:
            genotype_string += str(c) + ":"

        return genotype_string[:-1]

    def query(self):
        result = self.api.query_info_str_by_arch(self.genotype(), '90')
        return result

    def _loss(self, input, target):
        out, logits = self(input)
        loss = self._criterion(logits, target)
        if self.reg_type == 'kl' and self.optimizer_type == "drnas":
            loss += self._get_kl_reg()
        return loss, logits

    def _get_kl_reg(self):
        #TODO the following is a bit ugly-> refactor?
        cons_1 = (F.elu(self.alphas_level_1) + 1)
        cons_2 = (F.elu(self.alphas_level_2) + 1)
        cons_3 = (F.elu(self.alphas_level_3) + 1)
        cons_4 = (F.elu(self.alphas_level_4) + 1)
        cons_5 = (F.elu(self.alphas_level_5) + 1)
        q_1 = Dirichlet(cons_1)
        q_2 = Dirichlet(cons_2)
        q_3 = Dirichlet(cons_3)
        q_4 = Dirichlet(cons_4)
        q_5 = Dirichlet(cons_5)
        kl_reg = self.reg_scale * (torch.sum(kl_divergence(q_1, self.anchor_1)) + \
                                torch.sum(kl_divergence(q_2, self.anchor_2)) + \
                                torch.sum(kl_divergence(q_3, self.anchor_3)) + \
                                torch.sum(kl_divergence(q_4, self.anchor_4)) + \
                                torch.sum(kl_divergence(q_5, self.anchor_5)))
        return kl_reg

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format([
                nn.functional.softmax(a, dim=-1).cpu()
                for a in self.arch_parameters()
            ])

    def forward(self, inputs, arch_params=None):
        if arch_params == None:
            arch_params = self.sampler.sample_step(self.arch_parameters())
        arch_params = [arch_params[0]] + arch_params
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
        return out, logits


#test
#model = NATSSearchSpaceV2("gdas", genotype=CellStructure.str2structure('|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'), num_classes=10, criterion=nn.CrossEntropyLoss())#.cuda()
#img = torch.randn([64,3,32,32])#.cuda()
#print(model(img)[-1].shape)
#torch.save(model.state_dict(), "/work/dlclarge1/sukthank-transformer_search/GraViT-E/model_nats.path")
