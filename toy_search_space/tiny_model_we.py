from operations_we import *
from optimizers.optim_factory import get_mixop, get_sampler


class Net(nn.Module):

    def __init__(self, channels1, channels2, mlp_dim, optimizer):
        super(Net, self).__init__()
        self.mlp_dim = mlp_dim
        self.c1 = channels1
        self.c2 = channels2
        self.optimizer = optimizer
        self.sampler = get_sampler(optimizer)
        self.mixop1 = MixOP(optimizer)
        self.mixop2 = MixOP(optimizer)
        self.base_ops = [
            ConvOp5x5(1, self.c1),
            DWSConv7x7(1, self.c1),
            ConvMaxPool5x5(1, self.c1),
            ConvAvgPool5x5(1, self.c1),
            DilConv5x5(1, self.c1)
        ]
        self.oplist1 = torch.nn.ModuleList([
            self.base_ops[0],
            ConvOp3x3(self.base_ops[0]),
            ConvOp1x1(self.base_ops[0]), self.base_ops[1],
            DWSConv5x5(self.base_ops[1]),
            DWSConv3x3(self.base_ops[1]), self.base_ops[2],
            ConvMaxPool3x3(self.base_ops[2]),
            ConvMaxPool1x1(self.base_ops[2]), self.base_ops[3],
            ConvAvgPool3x3(self.base_ops[3]),
            ConvAvgPool1x1(self.base_ops[3]), self.base_ops[4],
            DilConv3x3(self.base_ops[4]),
            FactorizedReduce(1, self.c1)
        ])
        self.base_ops2 = [
            ConvOp5x5(self.c1, self.c2),
            DWSConv7x7(self.c1, self.c2),
            ConvMaxPool5x5(self.c1, self.c2),
            ConvAvgPool5x5(self.c1, self.c2),
            DilConv5x5(self.c1, self.c2)
        ]
        self.oplist2 = torch.nn.ModuleList([
            self.base_ops2[0],
            ConvOp3x3(self.base_ops2[0]),
            ConvOp1x1(self.base_ops2[0]), self.base_ops2[1],
            DWSConv5x5(self.base_ops2[1]),
            DWSConv3x3(self.base_ops2[1]), self.base_ops2[2],
            ConvMaxPool3x3(self.base_ops2[2]),
            ConvMaxPool1x1(self.base_ops2[2]), self.base_ops2[3],
            ConvAvgPool3x3(self.base_ops2[3]),
            ConvAvgPool1x1(self.base_ops2[3]),
            DilConv3x3_2(self.c1, self.c2),
            FactorizedReduce(self.c1, self.c2)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  #2x2 maxpool
        self.fc1 = nn.Linear(self.c2 * 3 * 3, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, 10)
        self.alphas1 = torch.nn.Parameter(1e-3 *
                                          torch.ones([len(self.oplist1)]))
        self.alphas2 = torch.nn.Parameter(1e-3 *
                                          torch.ones([len(self.oplist2)]))

    def get_model_params(self):
        param_list = []
        for n, p in self.named_parameters():
            if "alpha" not in n:
                param_list.append(p)
            else:
                continue
        return param_list

    def get_named_params(self):
        param_dict = {}
        for n, p in self.named_parameters():
            if "alpha" not in n:
                param_dict[n] = p
            else:
                continue
        for k in param_dict.keys():
            yield k, param_dict[k]

    def get_arch_params(self):
        return [self.alphas1, self.alphas2]

    def forward(self, x, tau):
        weights1 = self.sampler.sample_all_alphas([self.alphas1], tau)
        weights2 = self.sampler.sample_all_alphas([self.alphas2], tau)
        x = torch.nn.functional.relu(self.mixop1(x, self.oplist1,
                                                 weights1[-1]))
        x = torch.nn.functional.relu(self.mixop2(x, self.oplist2,
                                                 weights2[-1]))
        x = self.pool(x)
        x = x.view(-1, 3 * 3 * self.c2)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetDiscrete(nn.Module):

    def __init__(self, channels1, channels2, mlp_dim, idx1, idx2):
        super(NetDiscrete, self).__init__()
        self.mlp_dim = mlp_dim
        self.c1 = channels1
        self.c2 = channels2
        self.idx1 = idx1
        self.idx2 = idx2
        self.oplist1 = torch.nn.ModuleList([
            ConvOp5x5(1, self.c1),
            ConvOp3x3(1, self.c1),
            ConvOp1x1(1, self.c1),
            DWSConv7x7(1, self.c1),
            DWSConv5x5(1, self.c1),
            DWSConv3x3(1, self.c1),
            ConvMaxPool5x5(1, self.c1),
            ConvMaxPool3x3(1, self.c1),
            ConvMaxPool1x1(1, self.c1),
            ConvAvgPool5x5(1, self.c1),
            ConvAvgPool3x3(1, self.c1),
            ConvAvgPool1x1(1, self.c1),
            DilConv5x5(1, self.c1),
            DilConv3x3(1, self.c1),
            FactorizedReduce(1, self.c1)
        ])
        self.oplist2 = torch.nn.ModuleList([
            ConvOp5x5(self.c1, self.c2),
            ConvOp3x3(self.c1, self.c2),
            ConvOp1x1(self.c1, self.c2),
            DWSConv7x7(self.c1, self.c2),
            DWSConv5x5(self.c1, self.c2),
            DWSConv3x3(self.c1, self.c2),
            ConvMaxPool5x5(self.c1, self.c2),
            ConvMaxPool3x3(self.c1, self.c2),
            ConvMaxPool1x1(self.c1, self.c2),
            ConvAvgPool5x5(self.c1, self.c2),
            ConvAvgPool3x3(self.c1, self.c2),
            ConvAvgPool1x1(self.c1, self.c2),
            DilConv3x3(self.c1, self.c2),
            FactorizedReduce(self.c1, self.c2)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  #2x2 maxpool
        self.fc1 = nn.Linear(self.c2 * 3 * 3, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, 10)

    def get_model_params(self):
        param_list = []
        for n, p in self.named_parameters():
            if "alpha" not in n:
                param_list.append(p)
            else:
                continue
        return param_list

    def get_named_params(self):
        param_dict = {}
        for n, p in self.named_parameters():
            if "alpha" not in n:
                param_dict[n] = p
            else:
                continue
        for k in param_dict.keys():
            yield k, param_dict[k]

    def get_arch_params(self):
        return [self.alphas1, self.alphas2]

    def forward(self, x):
        x = torch.nn.functional.relu(self.oplist1[self.idx1](x))
        x = torch.nn.functional.relu(self.oplist2[self.idx2](x))
        x = self.pool(x)
        x = x.view(-1, 3 * 3 * self.c2)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MixOP(nn.Module):

    def __init__(self, optimizer):
        super(MixOP, self).__init__()
        self.mixop = get_mixop(optimizer)

    def forward(self, x, op_list, weights):
        out = self.mixop.forward(x, weights, op_list)
        return out
