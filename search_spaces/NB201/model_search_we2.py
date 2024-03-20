from copy import deepcopy
import torch.nn as nn
import torch
from optimizers.optim_factory import get_mixop, get_sampler
from torch.distributions.dirichlet import Dirichlet
from search_spaces.NB201.operations_we2 import OPS, ReLUConvBNSubSample, ResNetBasicblock, ReLUConvBNMixture
from search_spaces.base_model_search import SearchNetworkBase
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from search_spaces.NB201.genotypes import Structure
from nas_201_api import NASBench201API as API

class MixedOp(nn.Module):

  def __init__(self, mixop, primitives):
    super(MixedOp, self).__init__()
    self._ops = torch.nn.ModuleDict()
    self.mixop = mixop
    self.merge_indices = []
    i=0
    for primitive, op in primitives:
      if primitive == 'nor_conv_1x1':
        self._ops[primitive] = ReLUConvBNMixture(op, [1,3], 3) 
      elif "nor_conv_3x3" in primitive:
            self.merge_indices.append([i-1,i])
            i = i+1
            continue
      else:
        self._ops[primitive] = op
      i = i+1

  def forward(self, x, weights):
    return self.mixop.forward(x, weights, list(self._ops.values()),  merge_indices=self.merge_indices)

class NAS201SearchCell(nn.Module):
    def __init__(
        self,
        optimizer_type,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super(NAS201SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        self.mixop = get_mixop(optimizer_type)

        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    primitives = [
                        (op_name, OPS[op_name](C_in, C_out, stride, affine, track_running_stats))
                        for op_name in op_names
                    ]
                else:
                    primitives = [
                        (op_name, OPS[op_name](C_in, C_out, 1, affine, track_running_stats))
                        for op_name in op_names
                    ]
                self.edges[node_str] = MixedOp(self.mixop, primitives)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
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

class NASBench201SearchSpace(SearchNetworkBase):

    def __init__(
        self, optimizer_type, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, criterion, reg_type='l2', reg_scale=1e-3, path_to_benchmark=".", load_api=True, entangle_weights=True
    ):
        super(NASBench201SearchSpace, self).__init__()
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        self._C = C
        self._N = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        self._criterion = criterion
        self.num_classes = num_classes
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(
                    self.optimizer_type,
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
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self.num_edge = num_edge
        self.search_space = search_space
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        if load_api:
            self.api = API("/path/to/NAS-Bench-201-v1_0-e61699.pth")
        self._initialize_alphas()
        self._initialize_anchors()
        
    def _initialize_anchors(self):
        self.anchor = Dirichlet(torch.ones_like(self._arch_parameters[0]).to(DEVICE))

    def _initialize_alphas(self):
        self.arch_parameter = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(self.search_space))
        ) 
        self._arch_parameters = [self.arch_parameter]

    def _get_kl_reg(self):
       cons = (F.elu(self._arch_parameters[0]) + 1)
       q = Dirichlet(cons)
       p = self.anchor
       kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
       return kl_reg

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(self._arch_parameters[0], dim=-1).cpu()
            )

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_N}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self):
        genotypes = []
        alphas = torch.nn.functional.softmax(self._arch_parameters[0], dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = alphas[ self.edge2index[node_str] ]
                    op_name = self.op_names[ weights.argmax().item() ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return Structure(genotypes)

    def query(self):
        result = self.api.query_by_arch(self.genotype(),'200')
        return result

    def forward(self, inputs, alphas=None):
        alphas = self.sampler.sample_step(self._arch_parameters)[0] if alphas is None else alphas[0]

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, NAS201SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def _loss(self, input, target):
        _, logits = self(input)
        loss = self._criterion(logits, target)
        return loss, logits

    def _get_kl_reg(self):
        cons = (F.elu(self._arch_parameters[0]) + 1)
        q = Dirichlet(cons)
        p = self.anchor
        kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
        return kl_reg

    def new(self):

        model_new = NASBench201SearchSpace(
            self.optimizer_type,
            C=self._C,
            N=self._N,
            max_nodes=self.max_nodes,
            num_classes=self.num_classes,
            search_space=self.op_names,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            criterion=self._criterion,
            reg_type = self.reg_type,
            reg_scale = self.reg_scale,
            load_api=False
        ).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model_new