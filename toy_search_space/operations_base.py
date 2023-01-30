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

    def __init__(self, in_channels, out_channels):
        super(ConvOp3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvOp1x1(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvOp1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.conv(x))


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

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.bn(out)


class DWSConv5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWSConv5x5, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=5,
                                   stride=2,
                                   padding=2,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.bn(out)


class DWSConv3x3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWSConv3x3, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.bn(out)


class ConvMaxPool1x1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvMaxPool1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.pool(self.conv(x)))


class ConvMaxPool3x3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvMaxPool3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.pool(self.conv(x)))


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


class ConvAvgPool1x1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvAvgPool1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.pool(self.conv(x)))


class ConvAvgPool3x3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvAvgPool3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.bn(self.pool(self.conv(x)))


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


class DilConv3x3(nn.Module):

    def __init__(self, C_in, C_out):
        super(DilConv3x3, self).__init__()
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
