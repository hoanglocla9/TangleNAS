import torch
import torch.nn as nn
import torch.nn.functional as F
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]
class DilConvMixture(nn.Module):
    
  def __init__(self, op, kernel_size_list, kernel_max):
    super(DilConvMixture, self).__init__()
    self.op = op
    self.kernel_size_list = kernel_size_list
    self.kernel_max = kernel_max

  def forward(self, input):
    x, alphas_list = input
    x = self.op.op[0](x)
    weights_op1 = 0
    bias_op1 = 0
    for i, alpha in enumerate(alphas_list):
      kernel_size = self.kernel_size_list[i]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weight_curr = self.op.op[1].weight[:,:,start:end,start:end]
      weights_op1 += alpha * F.pad(weight_curr,(start,start,start,start),"constant", 0)
      if self.op.op[1].bias is not None:
        bias_op1 += alpha * self.op.op[1].bias
    x = F.conv2d(x,
                weight=weights_op1,
                bias=bias_op1 if self.op.op[1].bias is not None else None,
                stride=self.op.op[1].stride,
                padding=self.op.op[1].padding[0],
                dilation = self.op.op[1].dilation,
                groups = self.op.op[1].groups)  
    x = self.op.op[2](x)   
    x = self.op.op[3](x)
    return x

class SepConvMixture(nn.Module):
    
  def __init__(self, op, kernel_size_list, kernel_max):
    super(SepConvMixture, self).__init__()
    self.op = op
    self.kernel_size_list = kernel_size_list
    self.kernel_max = kernel_max

  def forward(self, input):
    x, alphas_list = input
    x = self.op.op[0](x)
    weights_op1 = 0
    bias_op1 = 0
    for i,alpha in enumerate(alphas_list):
      kernel_size = self.kernel_size_list[i]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weight_curr = self.op.op[1].weight[:,:,start:end,start:end]
      weights_op1 += alpha * F.pad(weight_curr,(start,start,start,start),"constant", 0)
      if self.op.op[1].bias is not None:
        bias_op1 += alpha * self.op.op[1].bias
    x = F.conv2d(x,
                weight=weights_op1,
                bias=bias_op1 if self.op.op[1].bias is not None else None,
                stride=self.op.op[1].stride,
                padding=self.op.op[1].padding[0],
                dilation = self.op.op[1].dilation,
                groups = self.op.op[1].groups)  
    x = self.op.op[2](x)   
    x = self.op.op[3](x)
    x = self.op.op[4](x)
    weights_op5 = 0
    bias_op5 = 0
    for i, alpha in enumerate(alphas_list):
      kernel_size = self.kernel_size_list[i]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weight_curr = self.op.op[5].weight[:,:,start:end,start:end]
      weights_op5 += alpha * F.pad(weight_curr,(start,start,start,start),"constant", 0)
      if self.op.op[5].bias is not None:
        bias_op5 += alpha * self.op.op[5].bias
    x = F.conv2d(x,
                weight=weights_op5,
                bias=bias_op5 if self.op.op[5].bias is not None else None,
                stride=self.op.op[5].stride,
                padding=self.op.op[5].padding[0],
                dilation = self.op.op[5].dilation,
                groups = self.op.op[5].groups) 
    x = self.op.op[6](x)
    x = self.op.op[7](x)
    return x 

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
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
      if self.op.padding[0]==2:
         x = F.conv2d(x,
                     weight=self.op.weight[:,:,1:(1+self.kernel_size),1:(1+self.kernel_size)],
                     bias=self.op.bias,
                     stride=self.op.stride,
                     padding=self.op.padding[0]-1,
                     groups = self.op.groups)
      else:
         x = F.conv2d(x,
                     weight=self.op.weight[:,:,1:(1+self.kernel_size),1:(1+self.kernel_size)],
                     bias=self.op.bias,
                     stride=self.op.stride,
                     padding=self.op.padding[0]-2,
                     dilation = self.op.dilation,
                     groups = self.op.groups)        
      return x


class DilConvSubSample(nn.Module):
    
  def __init__(self, layer, kernel_size):
    super(DilConvSubSample, self).__init__()
    self.op = nn.Sequential(
      layer.op[0],
      SubConv(layer.op[1],kernel_size),
      layer.op[2],
      layer.op[3])
  def forward(self, x):
    return self.op(x)

class SepConvSubSample(nn.Module):
    
  def __init__(self, layer, kernel_size):
    super(SepConvSubSample, self).__init__()
    self.op = nn.Sequential(
      layer.op[0],
      SubConv(layer.op[1],kernel_size),
      layer.op[2],
      layer.op[3],
      layer.op[4],
      SubConv(layer.op[5],kernel_size),
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
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
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
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out
