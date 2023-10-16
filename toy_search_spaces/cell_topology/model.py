from torch.profiler import profile, record_function, ProfilerActivity
from operations import OPS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
import itertools
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from utils import ReLUConvBN, Conv, Identity, FactorizedReduce  # noqa: E402

# this object will be needed to represent the discrete architecture extracted
# from the architectural parameters. See the method genotype() below.
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# operations set
PRIMITIVES = list(OPS.keys())  # operations set as list of strings


class MixedOp(nn.Module):
    """Base class for the mixed operation."""

    def __init__(self, C, stride):
        """
        :C: int; number of filters in each convolutional operation
        :stride: int; stride of the convolutional/pooling kernel
        """
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        # iterate thtough the operation set and append them to self._ops
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, alphas):
        """
        Compute the softmax of alphas and multiply that element-wise with the
        corresponding output of the operations.

        :x: torch.Tensor; input tensor
        :alphas: torch.Tensor; architectural parameters, either alphas_normal
        or alphas_reduce
        """
        # TODO: compute the softmax of the alphas parameter
        output = 0
        i = torch.argmax(alphas)
        return self._ops[i](x)


class Cell(nn.Module):
    """Base class for the cells in the search model."""

    def __init__(self, nodes, C_prev, C, reduction):
        """
        :nodes: int; number of intermediate nodes in the cell
        :C_prev: int; number of feature maps incoming to the cell
        :C: int; number of filters in each convolutional operation
        :reduction: bool; if it is a reduction or normal cell
        """
        super(Cell, self).__init__()
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
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, input, alphas):
        preprocessed_input = self.preprocess(input)
        alphas = alphas[1] if self.reduction else alphas[0]

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


class Network(nn.Module):
    """Base class for the search model (one-shot model)."""

    def __init__(self, device, nodes=2, num_channels=32):
        """
        :device: str; 'cuda' or 'cpu'
        :nodes: int; number of intermediate nodes in each cell
        """
        super(Network, self).__init__()
        self._nodes = nodes

        # the one-shot model we are going to use is composed of one reduction
        # cell followed by one normal cell and another reduction cell. The
        # architecture of the 2 reduction cells is the same (they share the
        # alpha_reduction parameter). However the weights of the corresponding
        # operations (convolutional filters) is different.
        reduction_cell_1 = Cell(nodes, 1, num_channels,
                                reduction=True)
        normal_cell = Cell(nodes, 2 * num_channels, num_channels,
                           reduction=False)
        reduction_cell_2 = Cell(nodes, 2 * num_channels, 2 * num_channels,
                                reduction=True)

        self.cells = nn.ModuleList([reduction_cell_1,
                                    normal_cell,
                                    reduction_cell_2])

        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(1152, 2*2*num_channels)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(2 * 2 * num_channels, 10)

        # initialize the architectural parameters to be equal. We also add a
        # tiny randomly sampled noise for numerical stability.

    def forward(self, input):
        x = input
        for i, cell in enumerate(self.cells):
            x = cell(x, self.arch_parameters)
            x = self.dropout(x)
        out = F.max_pool2d(x, 2)
        logits = self.classifier(self.dropout(self.relu(self.proj(out.view(out.size(0), -1)))))
        return logits

    def sample_exhaustively(self, device="cuda"):
        """
        Sample the architecture exhaustively by iterating through all possible
        combinations of the operations set.
        """
        k = sum(1 for i in range(self._nodes) for n in range(1 + i))
        num_ops = len(PRIMITIVES)
        sampled_archs = []
        archs = []
        for i in range(num_ops):
            arch_tensor = torch.zeros([num_ops]).to(device)
            arch_tensor[i] = 1
            archs.append(arch_tensor)
        for i in range(num_ops):
            for j in range(num_ops):
                for k in range(num_ops):
                    sampled_archs.append(torch.stack(
                        [archs[i], archs[j], archs[k]]))
        sampled_archs_cross_product = list(
            itertools.product(sampled_archs, sampled_archs))
        return sampled_archs_cross_product

    def sample_exhaustively_op_matrices(self, device="cpu"):
        """
        Sample all op matrices exhaustively
        """
        k = sum(1 for i in range(self._nodes) for n in range(1 + i))
        num_ops = len(PRIMITIVES)
        sampled_archs = []
        archs = []
        for i in range(num_ops):
            arch_tensor = torch.zeros([num_ops]).to(device)
            arch_tensor[i] = 1
            archs.append(arch_tensor)
        for i in range(num_ops):
            for j in range(num_ops):
                for k in range(num_ops):
                    sampled_archs.append(torch.stack(
                        [archs[i], archs[j], archs[k]]))
        op_matrices = []
        for i in range(len(sampled_archs)):
            op_matrix = torch.zeros([k+2, num_ops+2])
            op_matrix[0, 0] = 1
            op_matrix[-1, -1] = 1
            op_matrix[1:-1, 1:-1] = sampled_archs[i]
            op_matrices.append(op_matrix)
        op_matrices_norm_red = list(
            itertools.product(op_matrices, op_matrices))
        return op_matrices_norm_red

    def get_adjacency_matrix(self):
        adjacency = torch.Tensor([[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1, 1], [
                                 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        return adjacency

    def sample_arch(self, device):
        """
        Initialize the architectural parameters for the normal and reduction
        cells. The dimensions of each of these variables will be k x num_ops,
        where k is the number of edges in the cell and num_ops is the
        operation set size.
        """
        k = sum(1 for i in range(self._nodes) for n in range(1 + i))
        num_ops = len(PRIMITIVES)
        choice_normal = np.random.choice(num_ops, k)
        self.alphas_normal = torch.zeros(k, num_ops, device=device)
        self.alphas_normal[range(k), choice_normal] = 1
        choice_reduce = np.random.choice(num_ops, k)
        self.alphas_reduce = torch.zeros(k, num_ops, device=device)
        self.alphas_reduce[range(k), choice_reduce] = 1
        self.arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def init_arch_params(self, alphas_normal, alphas_reduce):
        self.alphas_normal = alphas_normal
        self.alphas_reduce = alphas_reduce
        self.arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def get_num_params(self, arch_params):
        """
        Get the number of parameters in  selected the model.
        """
        num_params = 0
        for n, p in self.named_parameters():
            if "cells" not in n:
                num_params += np.prod(p.size())
        j = 0
        for n, p in self.cells.named_parameters():
            if "preprocess" in n:
                num_params += np.prod(p.size())
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j]._ops)):
                if j % 2 == 0:
                    selected_op_id = torch.argmax(arch_params[0][i])
                    #print("Selected primitive", PRIMITIVES[selected_op_id])
                    selected_op = self.cells[j]._ops[i]._ops[selected_op_id]
                    for n, p in selected_op.named_parameters():
                        num_params += np.prod(p.size())
                else:
                    selected_op_id = torch.argmax(arch_params[1][i])
                    #print("Selected primitive", PRIMITIVES[selected_op_id])
                    selected_op = self.cells[j]._ops[i]._ops[selected_op_id]
                    for n, p in selected_op.named_parameters():
                        num_params += np.prod(p.size())

        return num_params

    def get_latency_from_string(self, s, sub_str="CPU time total: "):
        index = s.find(sub_str)  # find the index of the substring
        if index != -1:  # check if substring is found
            # extract content following substring
            content = s[index + len(sub_str):-3]
            # print(content)  # prints " jumps over the lazy dog"
            return content, s[-3:-1]
        else:
            print("Substring not found")

    def get_latency(self):
        """
        Get the latency of the selected model.
        """
        inputs = torch.randn([1, 1, 28, 28]).cuda()
        times_profiler_gpu = []
        for i in range(100):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    self(inputs)
            time, unit_gpu = self.get_latency_from_string(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=1), "CUDA time total: ")
            time = float(time)
            times_profiler_gpu.append(time)
        mean_cuda_time = np.mean(times_profiler_gpu)
        std_cuda_time = np.std(times_profiler_gpu)
        inputs = torch.randn([1, 1, 28, 28]).cpu()
        model_cpu = self.cpu()
        times_profiler_cpu = []
        for i in range(100):
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model_cpu(inputs)
            time, unit_cpu = self.get_latency_from_string(
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
            times_profiler_cpu.append(float(time))
        mean_cpu_time = np.mean(times_profiler_cpu)
        std_cpu_time = np.std(times_profiler_cpu)
        return mean_cuda_time, std_cuda_time, mean_cpu_time, std_cpu_time, unit_gpu, unit_cpu

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
