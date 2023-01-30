import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from search_spaces.DARTS.genotypes import PRIMITIVES, Genotype
from search_spaces.DARTS.operations_drnas import *
from optimizers.optim_factory import get_mixop, get_sampler
from search_spaces.base_model_search import SearchNetworkBase
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = int(num_channels // groups)
    # reshape
    x = x.view(batchsize, int(groups), channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class DARTSMixedOpWrapper(nn.Module):

    def __init__(self, C, stride, k, mixop):
        super(DARTSMixedOpWrapper, self).__init__()
        self.k = k
        self.C = C
        self._ops = torch.nn.ModuleDict()
        self.mp = nn.MaxPool2d(2, 2)
        self.mixop = mixop
        for primitive in PRIMITIVES:
            if primitive == 'sep_conv_3x3':
                self._ops[primitive] = SepConvSubSample(
                    self._ops['sep_conv_5x5'], 3)
            elif primitive == 'dil_conv_3x3':
                self._ops[primitive] = DilConvSubSample(
                    self._ops['dil_conv_5x5'], 3)
            else:
                op = OPS[primitive](int(C // self.k), stride, False)
                #if 'pool' in primitive:
                #op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops[primitive] = op

    def forward(self, x, weights):
        dim_2 = x.shape[1]
        xtemp = x[:, :int(dim_2 // self.k), :, :]
        xtemp2 = x[:, int(dim_2 // self.k):, :, :]
        #temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops) if not w == 0)
        #print(self._ops.values())
        temp1 = self.mixop.forward_progressive(xtemp, weights.to(x.device),
                                               self._ops.values())
        if self.k == 1:
            return temp1
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.k)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans

    def wider(self, k):
        self.k = k
        for op in self._ops.keys():
            self._ops[op].wider(self.C // k, self.C // k)
            #print(self._ops[op])


class Cell(nn.Module):

    def __init__(self, optimizer_type, steps, multiplier, C_prev_prev, C_prev,
                 C, reduction, reduction_prev, k):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.k = k
        self.mixop = get_mixop(optimizer_type)
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
                op = DARTSMixedOpWrapper(C, stride, self.k, self.mixop)
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

    def wider(self, k):
        self.k = k
        for op in self._ops:
            op.wider(k)


class DARTSSearchSpaceDrNAS(SearchNetworkBase):

    def __init__(self,
                 optimizer_type,
                 C,
                 num_classes,
                 layers,
                 criterion,
                 steps=4,
                 multiplier=4,
                 stem_multiplier=3,
                 k=6,
                 reg_type='l2',
                 reg_scale=1e-3):
        super(DARTSSearchSpaceDrNAS, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.k = k
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        self.api = None
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
            cell = Cell(self.optimizer_type, steps, multiplier, C_prev_prev,
                        C_prev, C_curr, reduction, reduction_prev, k)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self._initialize_anchors()

    def _initialize_anchors(self):
        self.anchor_normal = Dirichlet(
            torch.ones_like(self.alphas_normal).cuda())
        self.anchor_reduce = Dirichlet(
            torch.ones_like(self.alphas_reduce).cuda())

    def new(self):
        model_new = DARTSSearchSpaceDrNAS(self._C, self._num_classes,
                                          self._layers,
                                          self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def prune(self, x, num_keep, mask, reset=False):
        if not mask is None:
            x.data[~mask] -= 1000000
        src, index = x.topk(k=num_keep, dim=-1)
        if not reset:
            x.data.copy_(
                torch.zeros_like(x).scatter(dim=1, index=index, src=src))
        else:
            x.data.copy_(
                torch.zeros_like(x).scatter(dim=1,
                                            index=index,
                                            src=1e-3 * torch.randn_like(src)))
        mask = torch.zeros_like(x, dtype=torch.bool).scatter(
            dim=1, index=index, src=torch.ones_like(src, dtype=torch.bool))
        return mask

    def show_alphas(self):
        with torch.no_grad():
            logging.info('alphas normal :\n{:}'.format(
                self.process_step_matrix(self.alphas_normal, 'softmax',
                                         self.mask_normal).cpu()))
            logging.info('alphas reduce :\n{:}'.format(
                self.process_step_matrix(self.alphas_reduce, 'softmax',
                                         self.mask_reduce).cpu()))
            logging.info('concentration normal:\n{:}'.format(
                (F.elu(self.alphas_normal) + 1).cpu()))
            logging.info('concentration reduce:\n{:}'.format(
                (F.elu(self.alphas_reduce) + 1).cpu()))

    def pruning(self, num_keep):
        with torch.no_grad():
            self.mask_normal = self.prune(self.alphas_normal, num_keep,
                                          self.mask_normal)
            self.mask_reduce = self.prune(self.alphas_reduce, num_keep,
                                          self.mask_reduce)

    def wider(self, k):
        self.k = k
        for cell in self.cells:
            cell.wider(k)

    def process_step_vector(self, x, method, mask, tau=None):
        if method == 'softmax':
            output = F.softmax(x, dim=-1)
        elif method == 'dirichlet':
            output = torch.distributions.dirichlet.Dirichlet(F.elu(x) +
                                                             1).rsample()
        elif method == 'gumbel':
            output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)
        else:
            output = x
        if mask is None:
            return output
        else:
            output_pruned = torch.zeros_like(output)
            output_pruned[mask] = output[mask]
            output_pruned /= output_pruned.sum()
            assert (output_pruned[~mask] == 0.0).all()
            return output_pruned

    def process_step_matrix(self, x, method, mask, tau=None):
        weights = []
        if mask is None:
            for line in x:
                weights.append(
                    self.process_step_vector(line, method, None, tau))
        else:
            for i, line in enumerate(x):
                weights.append(
                    self.process_step_vector(line, method, mask[i], tau))
        return torch.stack(weights)

    def forward(self, input, arch_params=None):
        if arch_params == None:
            arch_params_sampled = self.sampler.sample_step(
                self._arch_parameters)
        else:
            arch_params_sampled = arch_params
        s0 = s1 = self.stem(input)

        weights_normal = self.process_step_matrix(arch_params_sampled[0], '',
                                                  self.mask_normal)
        weights_reduce = self.process_step_matrix(arch_params_sampled[1], '',
                                                  self.mask_reduce)
        if not self.mask_normal is None:
            assert (weights_normal[~self.mask_normal] == 0.0).all()
        if not self.mask_reduce is None:
            assert (weights_reduce[~self.mask_reduce] == 0.0).all()

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return out, logits

    def _loss(self, input, target):
        _, logits = self(input)
        loss = self._criterion(logits, target)
        if self.reg_type == 'kl' and self.optimizer_type == "drnas":
            loss += self._get_kl_reg()
        return loss, logits

    def _get_kl_reg(self):
        cons_normal = (F.elu(self.alphas_normal) + 1)
        cons_reduce = (F.elu(self.alphas_reduce) + 1)
        q_normal = Dirichlet(cons_normal)
        q_reduce = Dirichlet(cons_reduce)
        p_normal = self.anchor_normal
        p_reduce = self.anchor_reduce
        kl_reg = self.reg_scale * (torch.sum(kl_divergence(q_reduce, p_reduce)) + \
                                   torch.sum(kl_divergence(q_normal, p_normal)))
        return kl_reg

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
        self.mask_normal = None
        self.mask_reduce = None

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        # if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            self.process_step_matrix(self.alphas_normal, 'softmax',
                                     self.mask_normal).data.cpu().numpy())
        gene_reduce = _parse(
            self.process_step_matrix(self.alphas_reduce, 'softmax',
                                     self.mask_reduce).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal,
                            normal_concat=concat,
                            reduce=gene_reduce,
                            reduce_concat=concat)
        return genotype
