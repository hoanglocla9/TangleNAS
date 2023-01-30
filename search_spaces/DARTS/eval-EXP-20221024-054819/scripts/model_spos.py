import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES


def random_sample(steps):
    num_ops = len(PRIMITIVES)
    k = sum(1 for i in range(steps) for n in range(2 + i))
    sample = np.random.choice(num_ops, k)
    return sample


class ArchSampler(nn.Module):

    def __init__(self, C, stride):
        super(ArchSampler, self).__init__()
        self._ops = torch.nn.ModuleDict()
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

    def forward(self, x, op_id):
        #print(x.shape)
        #weights = weights.to(x.device)
        #return sum(w * op(x) for w, op in zip(weights, self._ops.values()))
        ops_list = list(self._ops.values())
        return ops_list[op_id](x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
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
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = ArchSampler(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, ops_selected):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, ops_selected[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self,
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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, ops_sampled):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = ops_sampled[1]
            else:
                weights = ops_sampled[0]
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target, ops_sampled):
        logits = self(input, ops_sampled)
        return self._criterion(logits, target)
