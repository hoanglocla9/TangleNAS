##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OPS", "RAW_OP_CLASSES", "ResNetBasicblock", "SearchSpaceNames"]
OPS_sub = {
    "none":
    lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    "avg_pool_3x3":
    lambda C_in, C_out, C_max, stride: POOLINGSub(C_in, C_out, C_max),
    "nor_conv_3x3_p1":
    lambda C_in, C_out, C_max, stride: ReLUConvBNSubP1(C_in, C_max),
    "nor_conv_3x3_p2":
    lambda C_in, C_out, C_max, stride: ReLUConvBNSubP2(C_out, C_max),
    "nor_conv_1x1_p1":
    lambda C_in, C_out, C_max, stride: ReLUConvBNSubP1(C_in, C_max),
    "nor_conv_1x1_p2":
    lambda C_in, C_out, C_max, stride: ReLUConvBNSubP2(C_out, C_max),
    "nor_conv_3x3":
    lambda C_in, C_out, C_max, stride: ReLUConvBNSub(C_in, C_out, C_max),
    "nor_conv_1x1":
    lambda C_in, C_out, C_max, stride: ReLUConvBNSub(C_in, C_out, C_max),
    "skip_connect_base":
    lambda C_in, C_out, C_max, stride: IdentityMasked(C_in),
    "skip_connect_p1":
    lambda C_in, C_out, C_max, stride: FactorizedReduceSubP1(C_in, C_max),
    "skip_connect_p2":
    lambda C_in, C_out, C_max, stride: FactorizedReduceSubP2(C_in, C_max),
    "skip_connect":
    lambda C_in, C_out, C_max, stride: IdentityMasked(C_in)
    if stride == 1 and C_in == C_out else FactorizedReduceSubSample(
        C_in, C_out, C_max),
}
OPS = {
    "none":
    lambda C_in, C_out, stride, affine, track_running_stats: Zero(
        C_in, C_out, stride),
    "avg_pool_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats),
    "max_pool_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "max", affine, track_running_stats),
    "nor_conv_7x7":
    lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7, 7),
        (stride, stride),
        (3, 3),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1":
    lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_5x5":
    lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (2, 2),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dil_sepc_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (2, 2),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "dil_sepc_5x5":
    lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (4, 4),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "skip_connect":
    lambda C_in, C_out, stride, affine, track_running_stats: FactorizedReduce(
        C_in, C_out, stride, affine, track_running_stats),
}

CONNECT_NAS_BENCHMARK = ["none", "skip_connect", "nor_conv_3x3"]
NAS_BENCH_201 = [
    "none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"
]
DARTS_SPACE = [
    "none",
    "skip_connect",
    "dua_sepc_3x3",
    "dua_sepc_5x5",
    "dil_sepc_3x3",
    "dil_sepc_5x5",
    "avg_pool_3x3",
    "max_pool_3x3",
]

SearchSpaceNames = {
    "connect-nas": CONNECT_NAS_BENCHMARK,
    "nats-bench": NAS_BENCH_201,
    "nas-bench-201": NAS_BENCH_201,
    "darts": DARTS_SPACE,
}


class FactorizedReduceSubSample(nn.Module):

    def __init__(self, C_in, C_out, C_max):
        super(FactorizedReduceSubSample, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.C_max = C_max
        self.bn_sampled = BatchNormSampled(C_out, C_max)

    def forward(self, x, op):
        if op.stride == 2:
            x = op.relu(x[:, :self.C_in, :, :])
            y = op.pad(x)
            out = torch.cat([
                torch.nn.functional.conv2d(
                    x,
                    weight=op.convs[0].weight[:self.C_out, :self.C_in, :, :],
                    bias=op.convs[0].bias[:self.C_out],
                    stride=op.convs[0].stride,
                    padding=op.convs[0].padding),
                torch.nn.functional.conv2d(
                    y[:, :, 1:, 1:],
                    weight=op.convs[1].weight[:self.C_out, :self.C_in, :, :],
                    bias=op.convs[1].bias[:self.C_out],
                    stride=op.convs[1].stride,
                    padding=op.convs[1].padding)
            ],
                            dim=1)
        else:
            bias = op.conv.bias
            if bias != None:
                bias = op.conv.bias[:self.C_out]
            #print(x.shape)
            #print(op.conv.weight[:self.C_out, :self.C_in, :, :].shape)
            out = torch.nn.functional.conv2d(
                x[:, :self.C_in, :, :],
                weight=op.conv.weight[:self.C_out, :self.C_in, :, :],
                bias=bias,
                stride=op.conv.stride,
                groups=op.conv.groups,
                padding=op.conv.padding)
        out = self.bn_sampled(out, op.bn, x.shape)
        return out


class FactorizedReduceSub(nn.Module):

    def __init__(self, C_in, C_out, C_max):
        super(FactorizedReduceSub, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.C_max = C_max
        self.part1 = FactorizedReduceSubP1(C_in, C_max)
        self.part2 = FactorizedReduceSubP2(C_out, C_max)

    def forward(self, x, op):
        out1, out2 = self.part1(x, op)
        out = self.part2(out1, out2, op)
        return out


class ResNetBasicblockSub(nn.Module):

    def __init__(self, C_in, C_out, C_max):
        super(ResNetBasicblockSub, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.C_max = C_max
        self.reluconvbn1 = ReLUConvBNSub(C_in, C_out, C_max)
        self.reluconvbn2 = ReLUConvBNSub(C_out, C_out, C_max)

    def forward(self, x, op):
        inputs = self.reluconvbn1(x, op.conv_a)
        inputs = self.reluconvbn2(inputs, op.conv_b)
        if op.downsample is not None:
            residual = op.downsample[0](x[:, :self.C_in, :, :])
            bias = op.downsample[1].bias
            if bias != None:
                bias = bias[:self.C_out]
            residual = torch.nn.functional.conv2d(
                residual,
                weight=op.downsample[1].weight[:self.C_out, :self.C_in, :, :],
                bias=bias,
                stride=op.downsample[1].stride,
                groups=op.downsample[1].groups,
                padding=op.downsample[1].padding)
        else:
            residual = x[:, :self.C_in, :, :]
        x = residual + inputs[:, :self.C_out, :, :]
        if self.C_out != self.C_max:
            x = F.pad(x, (0, 0, 0, 0, 0, self.C_max - self.C_out, 0, 0),
                      "constant", 0)
        return x


class BatchNormSampled(torch.nn.Module):

    def __init__(self, channel, max_channels):
        super().__init__()
        self.channel_choice = channel
        self.max_channels = max_channels

    def forward(self, x, bn, shape):
        #print(x.shape)
        #print(bn)
        if bn.training:
            bn_training = True
        else:
            bn_training = (bn.running_mean is None) and (bn.running_var is
                                                         None)
        if bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = bn.momentum

        if bn.training and bn.track_running_stats:
            assert bn.num_batches_tracked is not None
            bn.num_batches_tracked.add_(1)
            if bn.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / bn.num_batches_tracked.item(
                )
            else:  # use exponential moving average
                exponential_average_factor = bn.momentum
        #print(bn.weight)
        #print(bn.bias)
        
        if bn.running_mean is not None:
            running_mean = bn.running_mean[:self.channel_choice]
        else:
            running_mean = bn.running_mean
        if bn.running_var is not None:
            running_var = bn.running_var[:self.channel_choice]
        else:
            running_var = bn.running_var
        x = F.batch_norm(
            x[:, :self.channel_choice, :, :],
            # If buffers are not to be tracked,ensure that they won't be updated
            running_mean 
            if not bn.training or bn.track_running_stats else None,
            running_var
            if not bn.training or bn.track_running_stats else None,
            bn.weight[:self.channel_choice] if bn.weight is not None else bn.weight,
            bn.bias[:self.channel_choice] if bn.bias is not None else bn.bias,
            bn_training,
            exponential_average_factor,
            bn.eps,
        )
        #out_zero = torch.zeros(shape,
        #                      device=x.device)
        #print(x.shape)
        #out_zero[:, :self.channel_choice,:,:] = out
        #print(x.shape)
        if self.channel_choice != self.max_channels:
            x = F.pad(x, (0, 0, 0, 0, 0, self.max_channels - x.shape[1], 0, 0),
                      "constant", 0)
            #print(x.shape)
        return x


class ReLUConvBNSubP1(nn.Module):

    def __init__(self, C_in, C_max):
        super(ReLUConvBNSubP1, self).__init__()
        self.C_in = C_in
        self.C_max = C_max

    def forward(self, x, op):
        x = op.op[0](x[:, :self.C_in, :, :])
        #print(op.op[1].stride)
        x = torch.nn.functional.conv2d(
            x,
            weight=op.op[1].weight[:, :self.C_in, :, :],
            bias=op.op[1].bias,
            stride=op.op[1].stride,
            padding=op.op[1].padding)
        return x


class ReLUConvBNSubP2(nn.Module):

    def __init__(self, C_out, C_max):
        super(ReLUConvBNSubP2, self).__init__()
        self.C_out = C_out
        self.C_max = C_max
        self.bn = BatchNormSampled(C_out, C_max)

    def forward(self, x, op):
        x = self.bn(x, op.op[2], x.shape)
        return x


class ReLUConvBN(nn.Module):

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.track_running_stats = track_running_stats
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(C_out,
                           affine=affine,
                           track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(C_out,
                           affine=affine,
                           track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(C_in, C_out, kernel_size, 1, padding, dilation,
                            affine, track_running_stats)

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblockSubP1(nn.Module):

    def __init__(self, C_in, C_max):
        super(ResNetBasicblockSubP1, self).__init__()
        self.C_in = C_in
        self.C_max = C_max
        self.reluconvbn1p1 = ReLUConvBNSubP1(C_in, C_max)

    def forward(self, x, op):
        shape = x.shape
        inputs = self.reluconvbn1p1(x, op.conv_a)
        #inputs  = self.reluconvbn2p2(inputs,op.conv_b)
        if op.downsample is not None:
            residual = op.downsample[0](x[:, :self.C_in, :, :])
            bias = op.downsample[1].bias
            if bias != None:
                bias = bias[:self.C_out]
            residual = torch.nn.functional.conv2d(
                residual,
                weight=op.downsample[1].weight[:, :self.C_in, :, :],
                bias=bias,
                stride=op.downsample[1].stride,
                groups=op.downsample[1].groups,
                padding=op.downsample[1].padding)
        else:
            residual = x[:, :self.C_in, :, :]
        return inputs, residual


class ReLUConvBNSub(nn.Module):

    def __init__(self, C_in, C_out, C_max):
        super(ReLUConvBNSub, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.C_max = C_max
        self.bn = BatchNormSampled(C_out, C_max)

    def forward(self, x, op):
        shape = x.shape
        #print(x.shape)
        x = op.op[0](x[:, :self.C_in, :, :])
        bias = op.op[1].bias
        if bias != None:
            bias = op.op[1].bias[:self.C_out]
        #print(op.op[1].stride)
        x = torch.nn.functional.conv2d(
            x,
            weight=op.op[1].weight[:self.C_out, :self.C_in, :, :],
            bias=bias,
            stride=op.op[1].stride,
            padding=op.op[1].padding)
        #print(x.shape)
        x = self.bn(x, op.op[2], shape)
        #print("Relu Convbn shape",x.shape)
        return x


class ResNetBasicblockSubP2(nn.Module):

    def __init__(self, C_out, C_max):
        super(ResNetBasicblockSubP2, self).__init__()
        self.C_out = C_out
        self.C_max = C_max
        self.reluconvbn1 = ReLUConvBNSubP2(C_out, C_max)
        self.reluconvbn2 = ReLUConvBNSub(C_out, C_out, C_max)

    def forward(self, inputs, residual, op):
        #print(inputs.shape)
        inputs = self.reluconvbn1(inputs, op.conv_a)
        mask = torch.zeros([1, self.C_max, 1, 1], device=inputs.device)
        mask[:, :self.C_out, :, :] = 1
        inputs = inputs * mask
        inputs = self.reluconvbn2(inputs, op.conv_b)
        x = residual[:, :self.C_out, :, :] + inputs[:, :self.C_out, :, :]
        if self.C_out != self.C_max:
            x = F.pad(x, (0, 0, 0, 0, 0, self.C_max - self.C_out, 0, 0),
                      "constant", 0)
        return x


class ResNetBasicblock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 affine=True,
                 track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine,
                                 track_running_stats)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine,
                                 track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine,
                                         track_running_stats)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        #print(basicblock.shape)

        if self.downsample is not None:
            residual = self.downsample(inputs)
            #print(residual.shape)
        else:
            residual = inputs
            #print(residual.shape)
        return residual + basicblock


class POOLINGSub(nn.Module):

    def __init__(self, C_in, C_out, C_max):
        super(POOLINGSub, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.C_max = C_max
        self.op_sub = ReLUConvBNSub(C_in, C_out, C_max)
        if C_in == C_out:
            self.preprocess = False
        else:
            self.preprocess = True

    def forward(self, x, op):
        if self.preprocess:
            x = self.op_sub(x[:, :self.C_in, :, :], op.preprocess)
        else:
            x = x
        return op.op(x)


class POOLING(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 stride,
                 mode,
                 affine=True,
                 track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine,
                                         track_running_stats)
        if mode == "avg":
            self.op = nn.AvgPool2d(3,
                                   stride=stride,
                                   padding=1,
                                   count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class IdentityMasked(nn.Module):

    def __init__(self, C_in):
        super(IdentityMasked, self).__init__()
        self.C_in = C_in

    def forward(self, x, dummy):
        mask = torch.zeros([1, 64, 1, 1]).to(x.device)
        mask[:, :self.C_in, :, :] = 1
        return x * mask


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(
            **self.__dict__)


class FactorizedReduceSubP1(nn.Module):

    def __init__(self, C_in, C_max):
        super(FactorizedReduceSubP1, self).__init__()
        self.C_in = C_in
        #self.C_out = C_out
        self.C_max = C_max
        #self.bn_sampled = BatchNormSampled(C_out,C_max)

    def forward(self, x, op):
        if op.stride == 2:
            x = op.relu(x[:, :self.C_in, :, :])
            y = op.pad(x)
            conv1 = torch.nn.functional.conv2d(
                x,
                weight=op.convs[0].weight[:, :self.C_in, :, :],
                bias=op.convs[0].bias,
                stride=op.convs[0].stride,
                padding=op.convs[0].padding)
            conv2 = torch.nn.functional.conv2d(
                y[:, :, 1:, 1:],
                weight=op.convs[1].weight[:, :self.C_in, :, :],
                bias=op.convs[1].bias,
                stride=op.convs[1].stride,
                padding=op.convs[1].padding)
        else:
            bias = op.conv.bias
            #print(x.shape)
            #print(op.conv.weight[:self.C_out, :self.C_in, :, :].shape)
            conv1 = torch.nn.functional.conv2d(
                x[:, :self.C_in, :, :],
                weight=op.conv.weight[:, :self.C_in, :, :],
                bias=bias,
                stride=op.conv.stride,
                groups=op.conv.groups,
                padding=op.conv.padding)
            conv2 = None
        return conv1, conv2


class FactorizedReduceSubP2(nn.Module):

    def __init__(self, C_out, C_max):
        super(FactorizedReduceSubP2, self).__init__()
        self.C_out = C_out
        self.C_max = C_max
        self.bn_sampled = BatchNormSampled(C_out, C_max)
        self.C_outs = [C_out // 2, C_out - C_out // 2]

    def forward(self, x1, x2, op):
        out = None
        if x2 != None:
            out = torch.cat(
                [x1[:, :self.C_outs[0], :, :], x2[:, :self.C_outs[1], :, :]],
                dim=1)
            out = self.bn_sampled(out, op.bn, x1.shape)
        else:
            out = self.bn_sampled(x1, op.bn, x1.shape)
        return out


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in,
                              C_outs[i],
                              1,
                              stride=stride,
                              padding=0,
                              bias=not affine))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in,
                                  C_out,
                                  1,
                                  stride=stride,
                                  padding=0,
                                  bias=not affine)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(C_out,
                                 affine=affine,
                                 track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])],
                            dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(
            **self.__dict__)


# Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification, ICCV 2019
class PartAwareOp(nn.Module):

    def __init__(self, C_in, C_out, stride, part=4):
        super().__init__()
        self.part = 4
        self.hidden = C_in // 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv_list = nn.ModuleList()
        for i in range(self.part):
            self.local_conv_list.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(C_in, self.hidden, 1),
                    nn.BatchNorm2d(self.hidden, affine=True),
                ))
        self.W_K = nn.Linear(self.hidden, self.hidden)
        self.W_Q = nn.Linear(self.hidden, self.hidden)

        if stride == 2:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
        elif stride == 1:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
        else:
            raise ValueError("Invalid Stride : {:}".format(stride))

    def forward(self, x):
        batch, C, H, W = x.size()
        assert H >= self.part, "input size too small : {:} vs {:}".format(
            x.shape, self.part)
        IHs = [0]
        for i in range(self.part):
            IHs.append(min(H, int((i + 1) * (float(H) / self.part))))
        local_feat_list = []
        for i in range(self.part):
            feature = x[:, :, IHs[i]:IHs[i + 1], :]
            xfeax = self.avg_pool(feature)
            xfea = self.local_conv_list[i](xfeax)
            local_feat_list.append(xfea)
        part_feature = torch.cat(local_feat_list,
                                 dim=2).view(batch, -1, self.part)
        part_feature = part_feature.transpose(1, 2).contiguous()
        part_K = self.W_K(part_feature)
        part_Q = self.W_Q(part_feature).transpose(1, 2).contiguous()
        weight_att = torch.bmm(part_K, part_Q)
        attention = torch.softmax(weight_att, dim=2)
        aggreateF = torch.bmm(attention,
                              part_feature).transpose(1, 2).contiguous()
        features = []
        for i in range(self.part):
            feature = aggreateF[:, :, i:i + 1].expand(batch, self.hidden,
                                                      IHs[i + 1] - IHs[i])
            feature = feature.view(batch, self.hidden, IHs[i + 1] - IHs[i], 1)
            features.append(feature)
        features = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
        final_fea = torch.cat((x, features), dim=1)
        outputs = self.last(final_fea)
        return outputs


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x.mul_(mask)
    return x


# Searching for A Robust Neural Architecture in Four GPU Hours
class GDAS_Reduction_Cell(nn.Module):

    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, affine,
                 track_running_stats):
        super(GDAS_Reduction_Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine,
                                                track_running_stats)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, affine,
                                          track_running_stats)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, affine,
                                      track_running_stats)

        self.reduction = True
        self.ops1 = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    C,
                    C,
                    (1, 3),
                    stride=(1, 2),
                    padding=(0, 1),
                    groups=8,
                    bias=not affine,
                ),
                nn.Conv2d(
                    C,
                    C,
                    (3, 1),
                    stride=(2, 1),
                    padding=(1, 0),
                    groups=8,
                    bias=not affine,
                ),
                nn.BatchNorm2d(C,
                               affine=affine,
                               track_running_stats=track_running_stats),
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                nn.BatchNorm2d(C,
                               affine=affine,
                               track_running_stats=track_running_stats),
            ),
            nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    C,
                    C,
                    (1, 3),
                    stride=(1, 2),
                    padding=(0, 1),
                    groups=8,
                    bias=not affine,
                ),
                nn.Conv2d(
                    C,
                    C,
                    (3, 1),
                    stride=(2, 1),
                    padding=(1, 0),
                    groups=8,
                    bias=not affine,
                ),
                nn.BatchNorm2d(C,
                               affine=affine,
                               track_running_stats=track_running_stats),
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                nn.BatchNorm2d(C,
                               affine=affine,
                               track_running_stats=track_running_stats),
            ),
        ])

        self.ops2 = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.BatchNorm2d(C,
                               affine=affine,
                               track_running_stats=track_running_stats),
            ),
            nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.BatchNorm2d(C,
                               affine=affine,
                               track_running_stats=track_running_stats),
            ),
        ])

    @property
    def multiplier(self):
        return 4

    def forward(self, s0, s1, drop_prob=-1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        X0 = self.ops1[0](s0)
        X1 = self.ops1[1](s1)
        if self.training and drop_prob > 0.0:
            X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

        # X2 = self.ops2[0] (X0+X1)
        X2 = self.ops2[0](s0)
        X3 = self.ops2[1](s1)
        if self.training and drop_prob > 0.0:
            X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
        return torch.cat([X0, X1, X2, X3], dim=1)


# To manage the useful classes in this file.
RAW_OP_CLASSES = {"gdas_reduction": GDAS_Reduction_Cell}
