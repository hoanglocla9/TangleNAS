import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

OPS = {
  'sep_conv_3x3' : lambda C, stride: SepConv(C, C, 3, stride, 1, affine=False),
  'sep_conv_5x5' : lambda C, stride: SepConv(C, C, 5, stride, 2, affine=False),
  'dil_conv_3x3' : lambda C, stride: DilConv(C, C, 3, stride, 2, 2, affine=False),
  'dil_conv_5x5' : lambda C, stride: DilConv(C, C, 5, stride, 4, 2, affine=False)
}

class DilConvSuper(nn.Module):
    def __init__(self, op, kernel_size):
      super(DilConvSuper, self).__init__()
      self.op = op
      self.kernel_size = kernel_size
      self.kernel_max = 5

    def forward(self, x):
      x = self.op.op[0](x)
      start = (self.kernel_max-self.kernel_size)//2
      padding = self.op.op[1].padding[0]-(2*start)
      x = torch.nn.functional.conv2d(x, self.op.op[1].weight[:,:,start:(start+self.kernel_size),start:(start+self.kernel_size)], bias = self.op.op[1].bias, stride=self.op.op[1].stride, padding=padding, dilation=self.op.op[1].dilation, groups=self.op.op[1].groups)
      x = self.op.op[2](x)
      x = self.op.op[3](x)
      return x
    

class SepConvSuper(nn.Module):
  def __init__(self, op, kernel_size):
    super(SepConvSuper, self).__init__()
    self.op = op
    self.kernel_size = kernel_size
    self.kernel_max=5

  def forward(self, x):
    x = self.op.op[0](x)
    start = (self.kernel_max-self.kernel_size)//2
    padding = self.op.op[1].padding[0]-start
    x = torch.nn.functional.conv2d(x, self.op.op[1].weight[:,:,start:(start+self.kernel_size),start:(start+self.kernel_size)], bias = self.op.op[1].bias, stride=self.op.op[1].stride, padding=padding, dilation=self.op.op[1].dilation, groups=self.op.op[1].groups)
    x = self.op.op[2](x)
    x = self.op.op[3](x)
    x = self.op.op[4](x)
    start = (self.kernel_max-self.kernel_size)//2
    padding = self.op.op[5].padding[0]-start
    x = torch.nn.functional.conv2d(x, self.op.op[5].weight[:,:,start:(start+self.kernel_size),start:(start+self.kernel_size)], bias = self.op.op[5].bias, stride=self.op.op[5].stride, padding=padding, dilation=self.op.op[5].dilation, groups=self.op.op[5].groups)
    x = self.op.op[6](x)
    x = self.op.op[7](x)
    return x 


class DilConvMixture(nn.Module):
    
  def __init__(self, op, kernel_size_list, kernel_max):
    super(DilConvMixture, self).__init__()
    self.op = op
    self.kernel_size_list = kernel_size_list
    self.kernel_max = kernel_max
    self.kernel_size_min = min(kernel_size_list)
  
  def sample_weights(self, start, end):
    weight_sampled = self.op.op[1].weight[:,:,start:end,start:end]
    return weight_sampled

  def forward(self, input, alphas_list, use_argmax=False):
    x = input
    x = self.op.op[0](x)
    weights_op1 = 0
    #print("alphas_list", alphas_list)
    #print("kernel_size_list", self.kernel_size_list)
    if not use_argmax:
     for i, alpha in enumerate(alphas_list):
      kernel_size = self.kernel_size_list[i]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weight_curr = self.sample_weights(start, end)
      # pad weights
      weight_curr = F.pad(weight_curr,(start,start,start,start),"constant", 0)
      weights_op1 += alpha * weight_curr
      padding = self.op.op[1].padding[0]
    else:
      selected_alpha = torch.argmax(torch.tensor(alphas_list))
      kernel_size = self.kernel_size_list[torch.argmax(torch.tensor(alphas_list))]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weights_op1 = alphas_list[selected_alpha]*self.sample_weights(start, end)
      # pad weights
      if kernel_size == self.kernel_size_min:
        padding = self.op.op[1].padding[0]-(2*start)
      else:
        padding = self.op.op[1].padding[0]
    #print("weights_op1", weights_op1)
    x = F.conv2d(x,
                weight = weights_op1,
                bias = self.op.op[1].bias,
                stride = self.op.op[1].stride,
                padding = padding,
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
    self.kernel_size_min = min(kernel_size_list)
  
  def sample_weights(self, start, end, weight):
    weight_sampled = weight[:,:,start:end,start:end]
    # pad weights
    return weight_sampled
  
  def forward(self, input, alphas_list, use_argmax=False):
    x  = input
    x = self.op.op[0](x)
    weights_op1 = 0
    if not use_argmax:
     for i,alpha in enumerate(alphas_list):
      kernel_size = self.kernel_size_list[i]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weight_curr = self.sample_weights(start, end, self.op.op[1].weight)
      weight_curr = alphas_list[i]*weight_curr
      # pad weights
      weights_op1 += F.pad(weight_curr,(start,start,start,start),"constant", 0)
      padding = self.op.op[1].padding[0]
      #print("weights_op1", weights_op1)
    else:
      selected_alpha = torch.argmax(torch.tensor(alphas_list))
      kernel_size = self.kernel_size_list[torch.argmax(torch.tensor(alphas_list))]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weights_op1 = alphas_list[selected_alpha]*self.sample_weights(start, end, self.op.op[1].weight)
      if kernel_size == self.kernel_size_min:
        padding = self.op.op[1].padding[0]-start
      else:
        padding = self.op.op[1].padding[0]
      #print("weights_op1", weights_op1)
    x = F.conv2d(x,
                weight=weights_op1,
                bias = self.op.op[1].bias,
                stride=self.op.op[1].stride,
                padding=padding,
                dilation = self.op.op[1].dilation,
                groups = self.op.op[1].groups)  
    x = self.op.op[2](x)   
    x = self.op.op[3](x)
    x = self.op.op[4](x)
    weights_op5 = 0
    if not use_argmax:
     for i, alpha in enumerate(alphas_list):
      kernel_size = self.kernel_size_list[i]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weight_curr = self.sample_weights(start, end, self.op.op[5].weight)
      weight_curr = alphas_list[i]*weight_curr
      # pad weights
      weights_op5 += F.pad(weight_curr,(start,start,start,start),"constant", 0)
      padding = self.op.op[5].padding[0]
      #print("weights_op5", weights_op5)
    else:
      selected_alpha = torch.argmax(torch.tensor(alphas_list))
      kernel_size = self.kernel_size_list[torch.argmax(torch.tensor(alphas_list))]
      start = 0 + (self.kernel_max-kernel_size)//2
      end = start + kernel_size
      weights_op5 = alphas_list[selected_alpha]*self.sample_weights(start, end, self.op.op[5].weight)
      #print("weights_op5", weights_op5)
      # pad weights
      if kernel_size == self.kernel_size_min:
        padding = self.op.op[5].padding[0]-start
      else:
        padding = self.op.op[5].padding[0]
    x = F.conv2d(x,
                weight=weights_op5,
                bias= self.op.op[5].bias,
                stride=self.op.op[5].stride,
                padding=padding,
                dilation = self.op.op[5].dilation,
                groups = self.op.op[5].groups) 
    x = self.op.op[6](x)
    x = self.op.op[7](x)
    return x 
  
class ReLUConvBNMixture(nn.Module):
    def __init__(self, op, kernel_sizes):
        super(ReLUConvBNMixture, self).__init__()
        self.op = op

        assert len(kernel_sizes) == 2 # Assuming only 2 operations are entangled for now

        self.kernel_list = kernel_sizes
        self.kernel_max = max(kernel_sizes)

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias):
        alpha = weights[idx]

        kernel_size = self.kernel_list[idx]
        start = 0 + (self.kernel_max - kernel_size) // 2
        end = start + kernel_size
        weight_curr = self.op.op[1].weight[:, :, start:end, start:end]
        conv_weight += alpha * F.pad(weight_curr, (start, start, start, start), "constant", 0)

        if self.op.op[1].bias is not None:
            conv_bias += alpha * self.op.op[1].bias

        return conv_weight, conv_bias

    def forward(self, x, weights, use_argmax=False):
        x = self.op.op[0](x)

        conv_weight = 0
        conv_bias = 0

        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            conv_weight, conv_bias = self._compute_weight_and_bias(
                weights=weights,
                idx=argmax,
                conv_weight=conv_weight,
                conv_bias=conv_bias
            )
        else:
            for i, _ in enumerate(weights):
                conv_weight, conv_bias = self._compute_weight_and_bias(
                    weights=weights,
                    idx=i,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias
                )

        conv_bias = conv_bias if isinstance(conv_bias, torch.Tensor) else None

        x = F.conv2d(x,
                weight=conv_weight,
                bias=conv_bias,
                stride=self.op.op[1].stride,
                padding=self.op.op[1].padding[0],
                dilation = self.op.op[1].dilation,
                groups = self.op.op[1].groups)

        x = self.op.op[2](x)
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

class ConvAvgPool(nn.Module):
  
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ConvAvgPool, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
  
class ConvMaxPool(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
      super(ConvMaxPool, self).__init__()
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding),
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
  

'''import torch
from operations import OPS
a = torch.randn([32,4,14,14])
print(len(list(OPS.keys())))
stride = 1
if stride == 1:
  shape = a.shape
else:
  shape = torch.Size([a.shape[0],a.shape[1],a.shape[2]//stride,a.shape[3]//stride])
print(shape)
out = 0
for op in OPS.keys():
  operation = OPS[op](4,stride,False)
  out = out + operation(a)
  print("Operation",op)
  print(operation(a).shape)
  assert operation(a).shape == shape'''

# test mixture of operations dil conv
'''inp = torch.randn([2,1,32,32])
op = DilConv(1,1,5,1,2,2)
op.op[1].weight.data = torch.ones([1,1,5,5])
mixture_dil_conv = DilConvMixture(op,[3,5],5)
alphas = torch.tensor([0.5,0.5])
out = mixture_dil_conv(inp,alphas,use_argmax=True)
print(out.shape)
print(out)

# test mixture of operations sep conv
inp = torch.randn([2,1,32,32])
op = SepConv(1,1,5,1,2)
op.op[1].weight.data = torch.ones([1,1,5,5])
op.op[5].weight.data = torch.ones([1,1,5,5])
mixture_sep_conv = SepConvMixture(op,[3,5],5)
alphas = torch.tensor([0.5,0.5])
out = mixture_sep_conv(inp,alphas,use_argmax=False)
print(out.shape)
print(out)'''