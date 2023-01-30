import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvOp5x5(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvOp5x5, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=5,
                              stride=2,
                              padding=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvOp3x3(torch.nn.Module):

    def __init__(self, Conv5x5):
        super(ConvOp3x3, self).__init__()
        self.conv_base = Conv5x5.conv
        self.bn = Conv5x5.bn

    def forward(self, x):
        return self.bn(
            F.conv2d(x,
                     self.conv_base.weight[:, :, 1:4, 1:4],
                     bias=self.conv_base.bias,
                     stride=2,
                     padding=1))


class ConvOp1x1(torch.nn.Module):

    def __init__(self, Conv5x5):
        super(ConvOp1x1, self).__init__()
        self.conv_base = Conv5x5.conv
        self.bn = Conv5x5.bn

    def forward(self, x):
        return self.bn(
            F.conv2d(x,
                     self.conv_base.weight[:, :, 2:3, 2:3],
                     bias=self.conv_base.bias,
                     stride=2))


class DWSConv7x7(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWSConv7x7, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.bn(out)


class DWSConv5x5(nn.Module):

    def __init__(self, DWSConv7x7):
        super(DWSConv5x5, self).__init__()
        self.model = DWSConv7x7
        self.depthwise_base = DWSConv7x7.depthwise  #nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, groups=in_channels)
        self.pointwise_base = DWSConv7x7.pointwise  #(in_channels, out_channels, kernel_size=1)
        self.bn = DWSConv7x7.bn

    def forward(self, x):
        out = F.conv2d(x,
                       self.depthwise_base.weight[:, :, 1:6, 1:6],
                       bias=self.depthwise_base.bias,
                       stride=2,
                       padding=2,
                       groups=self.model.in_channels)
        out = F.conv2d(out,
                       self.pointwise_base.weight,
                       bias=self.pointwise_base.bias)
        return self.bn(out)


class DWSConv3x3(nn.Module):

    def __init__(self, DWSConv7x7):
        super(DWSConv3x3, self).__init__()
        self.model = DWSConv7x7
        self.depthwise_base = DWSConv7x7.depthwise  #nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, groups=in_channels)
        self.pointwise_base = DWSConv7x7.pointwise  #(in_channels, out_channels, kernel_size=1)
        self.bn = DWSConv7x7.bn

    def forward(self, x):
        out = F.conv2d(x,
                       self.depthwise_base.weight[:, :, 2:5, 2:5],
                       bias=self.depthwise_base.bias,
                       stride=2,
                       padding=1,
                       groups=self.model.in_channels)
        out = F.conv2d(out,
                       self.pointwise_base.weight,
                       bias=self.pointwise_base.bias)
        return self.bn(out)


class ConvMaxPool5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvMaxPool5x5, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.pool(self.conv(x)))


class ConvMaxPool3x3(nn.Module):

    def __init__(self, ConvMaxPool5x5):
        super(ConvMaxPool3x3, self).__init__()
        self.conv = ConvMaxPool5x5.conv  #nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.pool = ConvMaxPool5x5.pool  #nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn = ConvMaxPool5x5.bn  #nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(
            self.pool(
                F.conv2d(x,
                         self.conv.weight[:, :, 1:4, 1:4],
                         bias=self.conv.bias,
                         stride=1,
                         padding=1)))


class ConvMaxPool1x1(nn.Module):

    def __init__(self, ConvMaxPool5x5):
        super(ConvMaxPool1x1, self).__init__()
        self.conv = ConvMaxPool5x5.conv  #nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.pool = ConvMaxPool5x5.pool  #nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn = ConvMaxPool5x5.bn  #nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(
            self.pool(
                F.conv2d(x,
                         self.conv.weight[:, :, 2:3, 2:3],
                         bias=self.conv.bias,
                         stride=1)))


class ConvAvgPool5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvAvgPool5x5, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.pool(self.conv(x)))


class ConvAvgPool3x3(nn.Module):

    def __init__(self, ConvAvgPool5x5):
        super(ConvAvgPool3x3, self).__init__()
        self.conv = ConvAvgPool5x5.conv  #nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.pool = ConvAvgPool5x5.pool  #nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn = ConvAvgPool5x5.bn  #nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(
            self.pool(
                F.conv2d(x,
                         self.conv.weight[:, :, 1:4, 1:4],
                         bias=self.conv.bias,
                         stride=1,
                         padding=1)))


class ConvAvgPool1x1(nn.Module):

    def __init__(self, ConvAvgPool5x5):
        super(ConvAvgPool1x1, self).__init__()
        self.conv = ConvAvgPool5x5.conv  #nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.pool = ConvAvgPool5x5.pool  #nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn = ConvAvgPool5x5.bn  #nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(
            self.pool(
                F.conv2d(x,
                         self.conv.weight[:, :, 2:3, 2:3],
                         bias=self.conv.bias,
                         stride=1)))


class DilConv5x5(nn.Module):

    def __init__(self, C_in, C_out):
        super(DilConv5x5, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in,
                      C_in,
                      kernel_size=5,
                      stride=1,
                      padding=1,
                      dilation=4,
                      groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(C_out, affine=True))
        self.C_in = C_in

    def forward(self, x):
        return self.op(x)


class DilConv3x3(nn.Module):

    def __init__(self, DilConv5x5):
        super(DilConv3x3, self).__init__()
        self.model = DilConv5x5
        self.conv1 = DilConv5x5.op[0]
        self.conv2 = DilConv5x5.op[1]
        self.bn = DilConv5x5.op[2]

    def forward(self, x):
        out = F.conv2d(x,
                       self.conv1.weight[:, :, 1:4, 1:4],
                       bias=self.conv1.bias,
                       stride=2,
                       padding=2,
                       dilation=2,
                       groups=self.model.C_in)
        out = F.conv2d(out, self.conv2.weight, bias=self.conv2.bias, stride=1)
        return self.bn(out)


class DilConv3x3_2(nn.Module):

    def __init__(self, C_in, C_out):
        super(DilConv3x3_2, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in,
                      C_in,
                      kernel_size=3,
                      stride=2,
                      padding=2,
                      dilation=2,
                      groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(C_out, affine=True))

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
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
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)
