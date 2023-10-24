from copy import deepcopy
import torch.nn as nn
import torch
from optimizers.mixop.entangle import EntangledOp
from optimizers.optim_factory import get_mixop, get_sampler
from torch.distributions.dirichlet import Dirichlet
from search_spaces.NB201.operations import OPS, ReLUConvBNSubSample, ResNetBasicblock, ReLUConvBNMixture
from search_spaces.base_model_search import SearchNetworkBase
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
from search_spaces.NB201.genotypes import Structure
from nas_201_api import NASBench201API as API

from collections.abc import MutableMapping

class NB201MixedOpV2Wrapper(nn.Module):

    def __init__(self, mixop, primitives):
        super(NB201MixedOpV2Wrapper, self).__init__()
        self._ops = torch.nn.ModuleDict()
        self.mixop = mixop

        for primitive, op in primitives:
            if primitive == 'nor_conv_1x1':
                self._ops[primitive] = EntangledOp(op=None, name='nor_conv')
            elif primitive == 'nor_conv_3x3':
                op_mixture = ReLUConvBNMixture(op, kernel_sizes=(3, 1))
                self._ops[primitive] = EntangledOp(op=op_mixture, name='nor_conv')
            else:
                self._ops[primitive] = op
        #print(list(self._ops.values()))

    def forward(self, x, weights):
        return self.mixop.forward(x, weights, list(self._ops.values()))

    def __repr__(self):
        s = f'Operations {list(self._ops.keys())}'
        if self.entangle_weights == True:
            s += ' with weight entanglement VERSION 2'

        return s

    def register_backward_hook_on_entangled_op(self, op_name, hook_fn):
        ...

class NB201MixedOpWrapper(nn.Module):

    def __init__(self, mixop, primitives, entangle_weights=True):
        super(NB201MixedOpWrapper, self).__init__()
        self._ops = torch.nn.ModuleDict()
        self.mixop = mixop
        self.entangle_weights = entangle_weights

        if entangle_weights == True:
            self._init_entangled_ops(primitives)
        else:
            self._init_ops(primitives)

    def _init_entangled_ops(self, primitives):
        for primitive, op in primitives:
            if primitive == 'nor_conv_1x1':
                self._ops[primitive] = ReLUConvBNSubSample(
                    self._ops['nor_conv_3x3'], 1)
            else:
                self._ops[primitive] = op
  
    def _init_ops(self, primitives):
        for primitive, op in primitives:
            self._ops[primitive] = op

    def forward(self, x, weights):
        return self.mixop.forward(x, weights, list(self._ops.values()))

    def __repr__(self):
        s = f'Operations {list(self._ops.keys())}'
        if self.entangle_weights == True:
            s += ' with entangled weights'

        return s

    def register_backward_hook_on_entangled_op(self, op_name, hook_fn):
        #print(self._ops.keys())
        assert op_name in self._ops, f"{op_name} not found in NB201MixedOpWrapper._ops"

        reluconvbn_op = self._ops[op_name]
        conv = reluconvbn_op.op[1]
        conv.register_backward_hook(hook=hook_fn)

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
        entangle_weights=True,
        use_we_v2=False,
    ):
        super(NAS201SearchCell, self).__init__()
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        self.use_we_2 = use_we_v2
        self.mixop = get_mixop(optimizer_type, use_we_v2=use_we_v2)

        # Stats variables for tracking gradient contribution
        self.stats_total_grad_inputs = {}
        self.stats_backward_steps ={}
        self.stats_grad_norms = {}
        self.stats_last_grad_inputs = {}
        self.is_architect_step = False
       
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
                if use_we_v2:
                    self.edges[node_str] = NB201MixedOpV2Wrapper(self.mixop, primitives)
                else:
                    self.edges[node_str] = NB201MixedOpWrapper(self.mixop, primitives, entangle_weights=entangle_weights)
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

    def register_backward_hooks(self):

        def make_hook_fn(edge_name, op_name):
            def hook(module, grad_input, grad_output):
                if self.is_architect_step: # We're not interested in tracking gradients from the architect step
                    return

                for g_in in grad_input:
                    if g_in is not None:
                        g_in_shape = tuple(g_in.shape[-2:])
                        is_grad_wrt_kernel = False

                        if (g_in_shape == (1, 1) and op_name == 'nor_conv_1x1'):
                            is_grad_wrt_kernel = True
                        elif (g_in_shape == (3, 3) and op_name == 'nor_conv_3x3'):
                            is_grad_wrt_kernel = True

                        if is_grad_wrt_kernel == True:
                            key = f'{edge_name}::{op_name}'
                            g = g_in.cpu().detach().numpy()

                            if key not in self.stats_total_grad_inputs:
                                self.stats_total_grad_inputs[key] = g
                            else:
                                self.stats_total_grad_inputs[key] += g

                            self.stats_last_grad_inputs[key] = g

                            if key not in self.stats_backward_steps:
                                self.stats_backward_steps[key] = 1
                            else:
                                self.stats_backward_steps[key] += 1

                            mid = g_in.shape[-1] // 2
                            if key not in self.stats_grad_norms:
                                self.stats_grad_norms[key] = [torch.norm(g_in[:, :, mid, mid]).cpu().detach().numpy()]
                            else:
                                self.stats_grad_norms[key].append(torch.norm(g_in[:, :, mid, mid]).cpu().detach().numpy())

            return hook

        entangled_ops = ['nor_conv_1x1', 'nor_conv_3x3']

        for edge_name, op in self.edges.items():
            for op_name in entangled_ops:
                hook_fn = make_hook_fn(edge_name, op_name)
                op.register_backward_hook_on_entangled_op(op_name, hook_fn)

class NASBench201SearchSpace(SearchNetworkBase):

    def __init__(self,
                 optimizer_type,
                 C,
                 N,
                 max_nodes,
                 num_classes,
                 search_space,
                 affine,
                 track_running_stats,
                 criterion,
                 reg_type='l2',
                 reg_scale=1e-3,
                 path_to_benchmark = '/work/dlclarge1/sukthank-transformer_search/reproduce_oneshot/DrNAS/201-space/NAS-Bench-201-v1_0-e61699.pth',
                 entangle_weights=True,
                 initialize_api = True,
                 use_we_v2=False,
                 track_gradients=False):
        super(NASBench201SearchSpace, self).__init__()
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        self.op_names = deepcopy(list(reversed(search_space)))

        self._C = C
        self._N = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))
        self._criterion = criterion
        self.num_classes = num_classes
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        self.entangle_weights = entangle_weights
        self.use_we_v2 = use_we_v2
        self.track_gradients = track_gradients

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4
                                                            ] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
                zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(
                    self.optimizer_type,
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    self.op_names,
                    affine,
                    track_running_stats,
                    entangle_weights=entangle_weights,
                    use_we_v2=use_we_v2
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (num_edge == cell.num_edges and edge2index
                            == cell.edge2index), "invalid {:} vs. {:}.".format(
                                num_edge, cell.num_edges)

            if self.track_gradients is True \
                and self.entangle_weights is True \
                and isinstance(cell, NAS201SearchCell):
                cell.register_backward_hooks()

            self.cells.append(cell)
            C_prev = cell.out_dim
        self.num_edge = num_edge
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.type = "base"
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev),
                                     nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.path_to_benchmark = path_to_benchmark
        #if initialize_api:
        #   self.api = API(path_to_benchmark)
        self.api = None
        self._initialize_alphas()
        self._initialize_anchors()

    def _initialize_anchors(self):
        self.anchor = Dirichlet(
            torch.ones_like(self._arch_parameters[0]).to(DEVICE))

    def _initialize_alphas(self):
        self.arch_parameter = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(self.op_names)))
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
                nn.functional.softmax(self._arch_parameters[0], dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(i, len(self.cells),
                                                       cell.extra_repr())
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_N}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__)

    def genotype(self):
        genotypes = []
        alphas = torch.nn.functional.softmax(self._arch_parameters[0], dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = alphas[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def query(self):
        result = self.api.query_by_arch(self.genotype(), '200')
        return result

    def forward(self, inputs, alphas=None):
        alphas = self.sampler.sample_step(
            self._arch_parameters)[0] if alphas is None else alphas[0]
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
            search_space=list(reversed(self.op_names)),
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            criterion=self._criterion,
            reg_type=self.reg_type,
            reg_scale=self.reg_scale,
            path_to_benchmark=self.path_to_benchmark,
            entangle_weights = self.entangle_weights,
            initialize_api=False,
            use_we_v2=self.use_we_v2
        ).to(DEVICE)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model_new

    def get_saved_stats(self):
        stats = {}
        grad_norms_flat = {}

        def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
            items = []
            for k, v in d.items():
                new_key = str(parent_key) + sep + str(k) if parent_key else str(k)
                if isinstance(v, MutableMapping):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for idx, cell in enumerate(self.cells):

            if not isinstance(cell, NAS201SearchCell):
                continue

            stats[idx] = {}
            stats[idx]['avg_grad_inputs'] = {}

            for edge_op_name in cell.stats_total_grad_inputs.keys():
                stats[idx]['avg_grad_inputs'][edge_op_name] = cell.stats_total_grad_inputs[edge_op_name]/cell.stats_backward_steps[edge_op_name]

            stats[idx]['grad_norms'] = cell.stats_grad_norms
            stats[idx]['last_grad_inputs'] = cell.stats_last_grad_inputs
            grad_norms_flat[idx] = flatten_dict(cell.stats_grad_norms)

        stats['grad_norms_flat'] = flatten_dict(grad_norms_flat)
        return stats

    @property
    def is_architect_step(self):
        is_arch_steps = []
        for cell in self.cells:
            if isinstance(cell, NAS201SearchCell):
                is_arch_steps.append(cell.is_architect_step)

        assert (all(is_arch_steps) or all([not a for a in is_arch_steps]))
        return is_arch_steps[0]

    @is_architect_step.setter
    def is_architect_step(self, value):
        for cell in self.cells:
            if isinstance(cell, NAS201SearchCell):
                cell.is_architect_step = value

    def make_alphas_for_genotype(self, genotype):
        ops = [i for g in genotype.tolist("")[0] for i in g]
        alphas = torch.zeros_like(self.arch_parameters()[0])

        for idx, op in enumerate(ops):
            alphas[idx][self.op_names.index(op[0])] = 1.0

        return alphas

    def set_alphas_for_genotype(self, genotype):
        alphas = self.make_alphas_for_genotype(genotype)
        self.arch_parameters()[0].data = alphas.data

    def sample(self):
        self._initialize_alphas()
        genotype = self.genotype()
        self.set_alphas_for_genotype(genotype)
        return genotype

    def mutate(self, p=0.5):
        new_alphas = torch.zeros_like(self.arch_parameters()[0])

        for idx, row in enumerate(self.arch_parameters()[0]):
            if np.random.rand() < p:
                new_op = np.random.randint(0, new_alphas.shape[1])
                new_alphas[idx][new_op] = 1.0
            else:
                new_alphas[idx] = row.detach()

        self.arch_parameters()[0].data = new_alphas.data
        return self.genotype()

    def crossover(self, other_genotype, p=0.5):
        new_alphas = torch.zeros_like(self.arch_parameters()[0])
        other_alphas = self.make_alphas_for_genotype(other_genotype)

        for idx, row in enumerate(self.arch_parameters()[0]):
            if np.random.rand() < p:
                new_alphas[idx] = row.detach()
            else:
                new_alphas[idx] = other_alphas[idx].detach()

        self.arch_parameters()[0].data = new_alphas.data
        return self.genotype()

#tests
# from search_spaces.NB201.operations import NAS_BENCH_201 as NB201_SEARCH_SPACE
# model = NASBench201SearchSpace("darts_v1",16,5,4,10,NB201_SEARCH_SPACE,0,0,torch.nn.CrossEntropyLoss(),use_we_v2=True)
# model.arch_parameters()[0]=model.arch_parameters()[0]*2
# print(model.arch_parameters())
# print(model.genotype())
# input = torch.randn(2, 3, 32, 32)
# out, logits = model(input)
#torch.save(model.state_dict(), "/work/dlclarge1/sukthank-transformer_search/GraViT-E/model_nb201.path")
