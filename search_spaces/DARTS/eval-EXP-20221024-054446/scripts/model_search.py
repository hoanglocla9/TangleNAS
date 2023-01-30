import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from optimizers.optim_factory import get_mixop, get_sampler


class MixedOp(nn.Module):

    def __init__(self, C, stride, mixop):
        super(MixedOp, self).__init__()
        self._ops = torch.nn.ModuleDict()
        self.mixop = mixop
        for primitive in PRIMITIVES:
            if primitive == 'sep_conv_3x3':
                self._ops[primitive] = SepConvSubSample(
                    self._ops['sep_conv_5x5'], 3)
            elif primitive == 'dil_conv_3x3':
                self._ops[primitive] = DilConvSubSample(
                    self._ops['dil_conv_5x5'], 3)
            else:
                #op = OPS[primitive](C, stride, False)
                if 'pool' not in primitive:
                    self._ops[primitive] = OPS[primitive](C, stride, False)
                elif 'pool' in primitive:
                    self._ops[primitive] = nn.Sequential(
                        OPS[primitive](C, stride, False),
                        nn.BatchNorm2d(C, affine=False))

    def forward(self, x, weights):
        #print(x.shape)
        #weights = weights.to(x.device)
        #return sum(w * op(x) for w, op in zip(weights, self._ops.values()))
        return self.mixop.forward(x, weights, self._ops.values())


class Cell(nn.Module):

    def __init__(self, optimizer_type, steps, multiplier, C_prev_prev, C_prev,
                 C, reduction, reduction_prev):
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
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.mixop)
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


class Network(nn.Module):

    def __init__(self,
                 optimizer_type,
                 C,
                 num_classes,
                 layers,
                 criterion,
                 steps=4,
                 multiplier=4,
                 stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(optimizer_type, steps, multiplier, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self.optimizer_type, self._C, self._num_classes,
                            self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, tau_curr):
        arch_params_sampled = self.sampler.sample_all_alphas(
            self._arch_parameters, tau_curr)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(arch_params_sampled[1], dim=-1)
            else:
                weights = F.softmax(arch_params_sampled[0], dim=-1)
            #print(s0.shape)
            #print(s1.shape)
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target, tau_curr):
        logits = self(input, tau_curr)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(),
                                      requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(),
                                      requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

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
