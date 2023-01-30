#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import torch.nn as nn
import torch
from copy import deepcopy
from search_spaces.NB201.operations import OPS, ReLUConvBNSubSample, ResNetBasicblock
from search_spaces.NB201.operations import NAS_BENCH_201 as NB201_SEARCH_SPACE
from search_spaces.base_model import NetworkBase
from optimizers.optim_factory import get_mixop
# The macro structure for architectures in NAS-Bench-201


class MixedOpDiscretize(nn.Module):

    def __init__(self, primitives):
        super(MixedOpDiscretize, self).__init__()
        self._ops = torch.nn.ModuleDict()
        self.mixop = get_mixop("discrete")
        for primitive, op in reversed(primitives):
            if primitive == 'nor_conv_1x1':
                self._ops[primitive] = ReLUConvBNSubSample(
                    self._ops['nor_conv_3x3'], 1)
            else:
                self._ops[primitive] = op

    def forward(self, x, weights):
        return self.mixop.forward(x, weights, list(self._ops.values()))


class NAS201CellInfer(nn.Module):

    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super(NAS201CellInfer, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    primitives = [(op_name, OPS[op_name](C_in, C_out, stride,
                                                         affine,
                                                         track_running_stats))
                                  for op_name in op_names]
                else:
                    primitives = [(op_name, OPS[op_name](C_in, C_out, 1,
                                                         affine,
                                                         track_running_stats))
                                  for op_name in op_names]
                self.edges[node_str] = MixedOpDiscretize(primitives)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__)
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class NASBench201Model(NetworkBase):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine,
                 track_running_stats):
        super(NASBench201Model, self).__init__()
        self._C = C
        self._N = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))
        self.num_classes = num_classes
        self.affine = affine
        self.track_running_stats = track_running_stats
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4
                                                            ] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for _, (C_curr,
                reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201CellInfer(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (num_edge == cell.num_edges and edge2index
                            == cell.edge2index), "invalid {:} vs. {:}.".format(
                                num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self.num_edge = num_edge
        self.search_space = search_space
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev),
                                     nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()

    def before_epoch(self, drop_prob):
        pass

    def _initialize_alphas(self):
        self.arch_parameter = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(self.search_space)))
        self._arch_parameters = [self.arch_parameter]

    def forward(self, inputs, arch_params=None):
        alphas = self._arch_parameters[0]

        feature = self.stem(inputs)
        for _, cell in enumerate(self.cells):
            if isinstance(cell, NAS201CellInfer):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, None, logits


# How to inherit and forward prop
#model = NASBench201Inherit(16,5,4,10,NB201_SEARCH_SPACE,0,0)
#model.load_state_dict(torch.load("."))
#input = torch.randn([16,3,32,32])
#print(model(input)[0].shape)
