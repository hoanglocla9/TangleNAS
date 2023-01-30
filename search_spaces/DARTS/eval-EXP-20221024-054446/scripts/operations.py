import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'none':
    lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3':
    lambda C, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3':
    lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect':
    lambda C, stride, affine: Identity()
    if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3':
    lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5':
    lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7':
    lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3':
    lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5':
    lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7':
    lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C,
                  (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C,
                  (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in,
                      C_out,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in,
                      C_in,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        #print(self.op[1].weight.shape)

    def forward(self, x):
        return self.op(x)


class SubConv(nn.Module):

    def __init__(self, op, kernel_size):
        super(SubConv, self).__init__()
        #print(op.weight.shape)
        self.kernel_size = kernel_size
        self.op = op

    def forward(self, x):
        #print(self.weight_sub.shape)
        #print(x.device)
        #print(self.weight_sub.device)
        #print(self.op.stride)
        if self.op.padding[0] == 2:
            x = F.conv2d(x,
                         weight=self.op.weight[:, :, :self.kernel_size, :self.
                                               kernel_size],
                         bias=self.op.bias,
                         stride=self.op.stride,
                         padding=self.op.padding[0] - 1,
                         groups=self.op.groups)
        else:
            x = F.conv2d(x,
                         weight=self.op.weight[:, :, :self.kernel_size, :self.
                                               kernel_size],
                         bias=self.op.bias,
                         stride=self.op.stride,
                         padding=self.op.padding[0] - 2,
                         dilation=self.op.dilation,
                         groups=self.op.groups)
        return x


class DilConvSubSample(nn.Module):

    def __init__(self, layer, kernel_size):
        super(DilConvSubSample, self).__init__()
        self.op = nn.Sequential(layer.op[0], SubConv(layer.op[1], kernel_size),
                                layer.op[2], layer.op[3])

    def forward(self, x):
        return self.op(x)


class SepConvSubSample(nn.Module):

    def __init__(self, layer, kernel_size):
        super(SepConvSubSample, self).__init__()
        self.op = nn.Sequential(
            layer.op[0],
            SubConv(layer.op[1], kernel_size),
            layer.op[2],
            layer.op[3],
            layer.op[4],
            SubConv(layer.op[5], kernel_size),
            layer.op[6],
            layer.op[7],
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in,
                      C_in,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in,
                      C_in,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in,
                                C_out // 2,
                                1,
                                stride=2,
                                padding=0,
                                bias=False)
        self.conv_2 = nn.Conv2d(C_in,
                                C_out // 2,
                                1,
                                stride=2,
                                padding=0,
                                bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
