import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from search_spaces.DARTS.operations import *
from search_spaces.DARTS.genotypes import Genotype, PRIMITIVES
from optimizers.optim_factory import get_mixop, get_sampler
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
from search_spaces.base_model_search import SearchNetworkBase

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')


class DARTSMixedOpWrapper(nn.Module):

    def __init__(self, C, stride, mixop, entangle_weights=True):
        super(DARTSMixedOpWrapper, self).__init__()
        self._ops = torch.nn.ModuleDict()
        self.mixop = mixop
        self.entangle_weights = entangle_weights

        if entangle_weights == True:
            self._init_entangled_ops(C, stride)
        else:
            self._init_ops(C, stride)

    def _init_entangled_ops(self, C, stride):
        for primitive in PRIMITIVES:
            if primitive == 'sep_conv_3x3':
                self._ops[primitive] = SepConvSubSample(
                    self._ops['sep_conv_5x5'], 3)
            elif primitive == 'dil_conv_3x3':
                self._ops[primitive] = DilConvSubSample(
                    self._ops['dil_conv_5x5'], 3)
            else:
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops[primitive] = op

    def _init_ops(self, C, stride):
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops[primitive] = op

    def forward(self, x, weights):
        if sum(weights)==0:
            return 0
        else:
            return self.mixop.forward(x, weights.to(x.device),
                                  list(self._ops.values()))

    def __repr__(self):
        s = f'Operations {list(self._ops.keys())}'
        if self.entangle_weights == True:
            s += ' with entangled weights'

        return s

class Cell(nn.Module):

    def __init__(self, optimizer_type, steps, multiplier, C_prev_prev, C_prev,
                 C, reduction, reduction_prev, entangle_weights=True):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev,
                                          C,
                                          1,
                                          1,
                                          0,
                                          affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self.mixop = get_mixop(optimizer_type)
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        #print(self._steps)
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = DARTSMixedOpWrapper(C, stride, self.mixop, entangle_weights=entangle_weights)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class DARTSSearchSpace(SearchNetworkBase):

    def __init__(self,
                 optimizer_type,
                 C,
                 num_classes,
                 layers,
                 criterion,
                 steps=4,
                 multiplier=4,
                 stem_multiplier=3,
                 reg_type='l2',
                 reg_scale=1e-3,
                 entangle_weights=True):
        super(DARTSSearchSpace, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self.reg_scale = reg_scale
        self.reg_type = reg_type
        self._multiplier = multiplier
        self.api = None
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        self.entangle_weights = entangle_weights
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(optimizer_type, steps, multiplier, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev, entangle_weights)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()
        self._initialize_anchors()

    def _initialize_anchors(self):
        self.anchor_normal = Dirichlet(
            torch.ones_like(self.alphas_normal).to(DEVICE))
        self.anchor_reduce = Dirichlet(
            torch.ones_like(self.alphas_reduce).to(DEVICE))

    def new(self):
        model_new = DARTSSearchSpace(self.optimizer_type, self._C,
                                     self._num_classes, self._layers,
                                     self._criterion, entangle_weights=self.entangle_weights).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, alphas=None):
        arch_params_sampled = self.sampler.sample_step(
            self._arch_parameters) if alphas is None else alphas
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = arch_params_sampled[1]
            else:
                weights = arch_params_sampled[0]
            # print(s0.shape)
            # print(s1.shape)
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return out, logits  # TODO: return (out, logits)

    def _loss(self, input, target):
        _, logits = self(input)
        loss = self._criterion(logits, target)
        return loss, logits

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format([
                nn.functional.softmax(a, dim=-1).cpu()
                for a in self._arch_parameters
            ])

    def _get_kl_reg(self):
        cons_normal = (F.elu(self.alphas_normal) + 1)
        cons_reduce = (F.elu(self.alphas_reduce) + 1)
        q_normal = Dirichlet(cons_normal)
        q_reduce = Dirichlet(cons_reduce)
        p_normal = self.anchor_normal
        p_reduce = self.anchor_reduce
        kl_reg = self.reg_scale * (
            torch.sum(kl_divergence(q_reduce, p_reduce)) +
            torch.sum(kl_divergence(q_normal, p_normal)))
        return kl_reg

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 *
                                      torch.randn(k, num_ops).to(DEVICE),
                                      requires_grad=True)
        self.alphas_reduce = Variable(1e-3 *
                                      torch.randn(k, num_ops).to(DEVICE),
                                      requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                       if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal,
                            normal_concat=concat,
                            reduce=gene_reduce,
                            reduce_concat=concat)
        return genotype
