from toy_search_spaces.cell_topology.operations import OPS, DilConvMixture, SepConvMixture, SepConvSuper, DilConvSuper
from optimizers.mixop.entangle import EntangledOp
from toy_search_spaces.cell_topology.utils import ReLUConvBN
from optimizers.optim_factory import get_mixop, get_sampler
from torch.distributions.dirichlet import Dirichlet
from search_spaces.base_model_search import SearchNetworkBase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import sys
from os.path import dirname, abspath
from torch.autograd import Variable
import pickle
sys.path.append(dirname(dirname(abspath(__file__))))
# this object will be needed to represent the discrete architecture extracted
# from the architectural parameters. See the method genotype() below.
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# operations set
PRIMITIVES = list(OPS.keys())  # operations set as list of strings
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MixedOpWrapper(nn.Module):
    """Base class for the mixed operation."""

    def __init__(self, mixop, C, stride, entangle_weights=True, use_we_v2=True):
        """
        :mixop: nn.Module; the operation that combines the outputs of the operations
        :C: int; number of filters in each convolutional operation
        :stride: int; stride of the convolutional/pooling kernel
        """
        super(MixedOpWrapper, self).__init__()
        self.mixop = mixop
        self._ops = nn.ModuleDict()
        self.entangle_weights = entangle_weights
        self.use_we_v2 = use_we_v2
        if entangle_weights == True:
            self._init_entangled_ops(C, stride)
        else:
            self._init_ops(C, stride)

    def _init_entangled_ops(self, C, stride):
        if self.use_we_v2:
            for primitive in PRIMITIVES:
                # print(primitive)
                if primitive == 'sep_conv_5x5':
                    op_mixture = SepConvMixture(
                        OPS[primitive](C, stride), [3, 5], 5)
                    self._ops[primitive] = EntangledOp(
                        op=op_mixture, name='sep_conv')
                elif primitive == 'dil_conv_5x5':
                    op_mixture = DilConvMixture(
                        OPS[primitive](C, stride), [3, 5], 5)
                    self._ops[primitive] = EntangledOp(
                        op=op_mixture, name='dil_conv')
                elif primitive == 'sep_conv_3x3':
                    self._ops[primitive] = EntangledOp(
                        op=None, name='sep_conv')
                elif primitive == 'dil_conv_3x3':
                    self._ops[primitive] = EntangledOp(
                        op=None, name='dil_conv')
                else:
                    op = OPS[primitive](C, stride)
                    self._ops[primitive] = op

        else:
            super_op_dil_conv = OPS["dil_conv_5x5"](C, stride)
            super_op_sep_conv = OPS["sep_conv_5x5"](C, stride)
            for primitive in PRIMITIVES:
                if primitive == 'dil_conv_5x5':
                    self._ops[primitive] = DilConvSuper(super_op_dil_conv, 5)
                elif primitive == 'sep_conv_5x5':
                    self._ops[primitive] = SepConvSuper(super_op_sep_conv, 5)
                elif primitive == "dil_conv_3x3":
                    self._ops[primitive] = DilConvSuper(super_op_dil_conv, 3)
                elif primitive == "sep_conv_3x3":
                    self._ops[primitive] = SepConvSuper(super_op_sep_conv, 3)
                else:
                    op = OPS[primitive](C, stride)
                    self._ops[primitive] = op

    def _init_ops(self, C, stride):
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride)
            self._ops[primitive] = op

    def forward(self, x, alphas):
        """
        Compute the softmax of alphas and multiply that element-wise with the
        corresponding output of the operations.

        :x: torch.Tensor; input tensor
        :alphas: torch.Tensor; architectural parameters, either alphas_normal
        or alphas_reduce
        """
        out = self.mixop.forward(
                    x, alphas.to(x.device), list(self._ops.values()))
        return out


class Cell(nn.Module):
    """Base class for the cells in the search model."""

    def __init__(self, optimizer_type, nodes, C_prev, C, reduction, entangle_weights=True, use_we_v2=True):
        """
        :nodes: int; number of intermediate nodes in the cell
        :C_prev: int; number of feature maps incoming to the cell
        :C: int; number of filters in each convolutional operation
        :reduction: bool; if it is a reduction or normal cell
        """
        super(Cell, self).__init__()
        self.mixop = get_mixop(optimizer_type, use_we_v2=use_we_v2)
        self.reduction = reduction
        # this preprocessing operation is added to keep the dimensions of the
        # tensors going to the intermediate nodes the same.
        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)
        self._nodes = nodes

        self._ops = nn.ModuleList()
        # iterate throughout each edge of the cell and create a MixedOp
        for i in range(self._nodes):
            for j in range(1 + i):
                stride = 2 if reduction and j < 1 else 1
                op = MixedOpWrapper(self.mixop, C, stride, entangle_weights=entangle_weights,
                                    use_we_v2=use_we_v2)
                self._ops.append(op)

    def forward(self, input, alphas):
        preprocessed_input = self.preprocess(input)

        states = [preprocessed_input]
        offset = 0
        for i in range(self._nodes):
            s = sum(self._ops[offset + j](h, alphas[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # concatenate the outputs of only the intermediate nodes to form the
        # output node.
        out = torch.cat(states[-self._nodes:], dim=1)
        return out

    def get_weights(self):
        return [op.weights for op in self._ops]


class ToyCellSearchSpace(SearchNetworkBase):
    """Base class for the search model (one-shot model)."""

    def __init__(self, optimizer_type, num_classes, criterion, api=None, max_nodes=2, num_channels=32, entangle_weights=True, use_we_v2=True):
        """
        :nodes: int; number of intermediate nodes in each cell
        """
        super(ToyCellSearchSpace, self).__init__()
        assert not (
            entangle_weights is False and use_we_v2 is True), "Cannot use we_v2 when weight entanglement is disabled!"
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        self._nodes = max_nodes
        self.num_channels = num_channels
        self._num_classes = num_classes
        self._criterion = criterion
        self._num_classes = num_classes
        self.max_nodes = max_nodes
        self.api = api
        self.entangle_weights = entangle_weights
        self.type = 'toy'
        self.use_we_v2 = use_we_v2
        reduction_cell_1 = Cell(optimizer_type, max_nodes, 1, num_channels, reduction=True,
                                entangle_weights=entangle_weights, use_we_v2=use_we_v2)
        normal_cell = Cell(
            optimizer_type, max_nodes, 2 * num_channels, num_channels, reduction=False,  entangle_weights=entangle_weights, use_we_v2=use_we_v2)
        reduction_cell_2 = Cell(
            optimizer_type, max_nodes, 2 * num_channels, 2 * num_channels, reduction=True,  entangle_weights=entangle_weights, use_we_v2=use_we_v2)

        self.cells = nn.ModuleList([
            reduction_cell_1,
            normal_cell,
            reduction_cell_2
        ])

        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(1152, 2 * 2 * num_channels)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(2 * 2 * num_channels, num_classes)

        if self.api is not None:
            with open(self.api, "rb") as f:
                self.benchmark = pickle.load(f)

        self._initialize_alphas()

    def add_betas_to_arch_params(self):
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce
        ]

    def forward(self, input, alphas=None):
        alphas = self.sampler.sample_step(
            self._arch_parameters) if alphas is None else alphas
        x = input

        for i, cell in enumerate(self.cells):
            arch_weights = alphas[1] if cell.reduction else alphas[0]

            x = cell(x, arch_weights)
            x = self.dropout(x)

        out = F.max_pool2d(x, 2)
        out = self.proj(out.view(out.size(0), -1))
        out = self.dropout(self.relu(out))
        logits = self.classifier(out)

        return out, logits

    def query(self):
        genotype = self.genotype()
        return self.benchmark[str(genotype)]

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format([
                nn.functional.softmax(a, dim=-1).cpu()
                for a in self._arch_parameters
            ])

    def new(self):
        model_new = ToyCellSearchSpace(self.optimizer_type, self._num_classes, self._criterion, api=self.api, max_nodes=self.max_nodes,
                                       num_channels=self.num_channels, entangle_weights=self.entangle_weights, use_we_v2=self.use_we_v2)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, input, target):
        _, logits = self(input)
        loss = self._criterion(logits, target)
        return loss, logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._nodes) for n in range(1 + i))

        self.alphas_normal = torch.nn.Parameter(
            1e-3 * torch.randn(k, len(PRIMITIVES), requires_grad=False), requires_grad=True)

        self.alphas_reduce = torch.nn.Parameter(
            1e-3 * torch.randn(k, len(PRIMITIVES), requires_grad=False), requires_grad=True)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def genotype(self, alphas=None):
        """
        Method for getting the discrete architecture, represented as a Genotype
        object from the DARTS search model.
        """

        def _parse(alphas):
            gene = []
            n = 1
            start = 0
            for i in range(self._nodes):
                end = start + n
                W = alphas[start:end].copy()
                for j in range(i+1):
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        if alphas is None:
            gene_normal = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce = _parse(
                F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        else:
            gene_normal = _parse(
                F.softmax(alphas[0], dim=-1).data.cpu().numpy())
            gene_reduce = _parse(
                F.softmax(alphas[1], dim=-1).data.cpu().numpy())

        concat = range(1, self._nodes + 1)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype


'''if __name__ == "__main__":

    optimizers = ["darts_v1", "darts_v2", "gdas", "drnas"]

    for optimizer in optimizers:
        model_search = ToyCellSearchSpace(optimizer, 10, torch.nn.CrossEntropyLoss())
        if optimizer == "gdas":
            model_search.sampler.set_taus(0.1,10)
            model_search.sampler.set_total_epochs(100)
            model_search.sampler.before_epoch()
        x = torch.randn(3, 1, 28, 28)
        out = model_search(x)
        print(optimizer, out[0].shape, out[1].shape)'''
