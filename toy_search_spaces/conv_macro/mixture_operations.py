import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSuperFixedInCh(nn.Module):
    def __init__(self, base_op, channel_choice, kernel_choice):
        super(ConvSuperFixedInCh,self).__init__()
        
        self.op = base_op
        self.channel_choice = channel_choice
        self.kernel_choice = kernel_choice

    def get_weights_and_bias(self):
        weight = self.op.weight[:self.channel_choice,:,0:self.kernel_choice,0:self.kernel_choice]
        if self.op.bias is not None:
            bias = self.op.bias[:self.channel_choice]
        else:
            bias = None
        return weight, bias
    
    def padding(self):
        if self.kernel_choice == 3:
            padding = 1
        elif self.kernel_choice == 5:
            padding = 2
        else:
            padding = 3
        return padding
    
    def forward(self, x):
        weight, bias = self.get_weights_and_bias()
        padding = self.padding()
        x = torch.nn.functional.conv2d(x, weight, bias, padding=padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        # pad output to max channel choice
        x = torch.nn.functional.pad(x, (0,0,0,0,0,self.op.out_channels-self.channel_choice,0,0))
        return x
    
class ConvSuperFixedOutCh(nn.Module):
    def __init__(self, base_op, channel_choice):
        super(ConvSuperFixedOutCh,self).__init__()
        
        self.op = base_op
        self.channel_choice = channel_choice

    def get_weights_and_bias(self):
        weight = self.op.weight[:,:self.channel_choice,:,:]
        if self.op.bias is not None:
            bias = self.op.bias[:]
        else:
            bias = None
        return weight, bias
    
    def padding(self):
        return 2
    
    def forward(self, x):
        weight, bias = self.get_weights_and_bias()
        padding = self.padding()
        x = torch.nn.functional.conv2d(x[:,:self.channel_choice,:,:], weight, bias, padding=padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        # pad output to max channel choice
        return x

class ConvSuper(nn.Module):
    def __init__(self, base_op, channel_choice_in, channel_choice_out, kernel_choice):
        super(ConvSuper,self).__init__()
        
        self.op = base_op
        self.channel_choice_in = channel_choice_in
        self.channel_choice_out = channel_choice_out
        self.kernel_choice = kernel_choice

    def get_weights_and_bias(self):
        weight = self.op.weight[:self.channel_choice_out,:self.channel_choice_in,0:self.kernel_choice,0:self.kernel_choice]
        if self.op.bias is not None:
            bias = self.op.bias[:self.channel_choice_out]
        else:
            bias = None
        return weight, bias
    
    def padding(self):
        if self.kernel_choice == 3:
            padding = 1
        elif self.kernel_choice == 5:
            padding = 2
        else:
            padding = 3
        return padding

    
    def forward(self, x):
        weight, bias = self.get_weights_and_bias()
        padding = self.padding()
        x = torch.nn.functional.conv2d(x[:,:self.channel_choice_in,:,:], weight, bias, padding=padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        # pad output to max channel choice
        x = torch.nn.functional.pad(x, (0,0,0,0,0,self.op.out_channels-self.channel_choice_out,0,0))
        return x

class ConvMixtureFixedInCh(torch.nn.Module):
    def __init__(self, base_op, choices_channel, choices_kernel):
        super(ConvMixtureFixedInCh,self).__init__()
        
        self.op = base_op
        self.choices_channel = choices_channel
        self.choices_kernel = choices_kernel
        self.max_kernel = max(choices_kernel)
        self.max_channel = max(choices_channel)

    def get_weights_and_bias(self,channel_choice,kernel_choice,start):
        weight = self.op.weight[:channel_choice,:,start:start+kernel_choice,start:start+kernel_choice]
        if self.op.bias is not None:
            bias = self.op.bias[:channel_choice]
        else:
            bias = None
        return weight, bias
    
    def get_padding(self,kernel_choice):
        if kernel_choice == 3:
            padding = 1
        elif kernel_choice == 5:
            padding = 2
        else:
            padding = 3
        return padding

    def forward(self, x, weights, use_argmax=False):
        if use_argmax==True:
            #print(weights)
            weights_max_id = torch.argmax(torch.tensor(weights))
            channel_id = torch.div(weights_max_id, len(self.choices_kernel), rounding_mode='trunc')
            kernel_id = weights_max_id % len(self.choices_kernel)
            #print("channel: ", self.choices_channel[channel_id])
            #print("kernel: ", self.choices_kernel[kernel_id])
            start = (self.max_kernel - self.choices_kernel[kernel_id])//2
            weight, bias = self.get_weights_and_bias(self.choices_channel[channel_id],self.choices_kernel[kernel_id],start)
            # pad weights and bias to max channel choice and max kernel choice
            weight  = weights[weights_max_id]*weight #torch.nn.functional.pad(weight, (start,start,start,start,0,0,0,self.max_channel-self.choices_channel[channel_id]))
            bias = weights[weights_max_id]*bias #torch.nn.functional.pad(bias, (0,self.max_channel-self.choices_channel[channel_id]))
            #print(weight)
            #print(bias)
            padding = self.get_padding(self.choices_kernel[kernel_id])
            x = torch.nn.functional.conv2d(x, weight, bias, padding=padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        else:
            weight_mix = 0
            bias_mix = 0
            arch_param_id = 0
            for i in range(len(self.choices_channel)):
                for j in range(len(self.choices_kernel)):
                    start = (self.max_kernel - self.choices_kernel[j])//2
                    weight, bias = self.get_weights_and_bias(self.choices_channel[i],self.choices_kernel[j],start)
                    # pad weights and bias to max channel choice and max kernel choice
                    weight  = torch.nn.functional.pad(weight, (start,start,start,start,0,0,0,self.max_channel-self.choices_channel[i]))
                    bias = torch.nn.functional.pad(bias, (0,self.max_channel-self.choices_channel[i]))
                    weight_mix += weights[arch_param_id]*weight 
                    bias_mix += weights[arch_param_id]*bias
                    arch_param_id += 1
            #print(weight_mix)
            #print(bias_mix)
            x = torch.nn.functional.conv2d(x, weight_mix, bias_mix, padding=self.op.padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        return x
    
class ConvMixtureFixedOutCh(torch.nn.Module):
    def __init__(self, base_op, choices_channel):
        super(ConvMixtureFixedOutCh,self).__init__()
        
        self.op = base_op
        self.choices_channel = choices_channel
        self.max_channel = max(choices_channel)

    def get_weights_and_bias(self,channel_choice):
        weight = self.op.weight[:,:channel_choice,:,:]
        if self.op.bias is not None:
            bias = self.op.bias[:]
        else:
            bias = None
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax==True:
            #print(weights)
            channel_id = torch.argmax(torch.tensor(weights))
            weight, bias = self.get_weights_and_bias(self.choices_channel[channel_id])
            # pad weights and bias to max channel choice and max kernel choice
            weight  = weights[channel_id]*weight#torch.nn.functional.pad(weight, (0,0,0,0,0,self.max_channel-self.choices_channel[channel_id],0,0))
            #print(weight)
            x = torch.nn.functional.conv2d(x, weight, bias, padding=self.op.padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        else:
            weight_mix = 0
            bias_mix = 0
            arch_param_id = 0
            for i in range(len(self.choices_channel)):
                # pad weights and bias to max channel choice and max kernel choice
                weight, bias = self.get_weights_and_bias(self.choices_channel[i])
                weight  = torch.nn.functional.pad(weight, (0,0,0,0,0,self.max_channel-self.choices_channel[i],0,0))
                weight_mix += weights[arch_param_id]*weight 
                bias_mix  = bias
                arch_param_id += 1
            #print(weight_mix)
            #print(bias_mix)
            x = torch.nn.functional.conv2d(x, weight_mix, bias_mix, padding=self.op.padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        return x

class ConvMixture(torch.nn.Module):
    def __init__(self, base_op, choices_channel_in, choices_channel_out, choices_kernel):
        super(ConvMixture,self).__init__()
        
        self.op = base_op
        self.choices_channel_in = choices_channel_in
        self.choices_channel_out = choices_channel_out
        self.choices_kernel = choices_kernel
        self.max_kernel = max(choices_kernel)
        self.max_channel_out = max(choices_channel_out)
        self.max_channel_in = max(choices_channel_in)

    def get_weights_and_bias(self, channel_choice_in, channel_choice_out, kernel_choice, start):
        weight = self.op.weight[:channel_choice_out,:channel_choice_in,start:start+kernel_choice,start:start+kernel_choice]
        if self.op.bias is not None:
            bias = self.op.bias[:channel_choice_out]
        else:
            bias = None
        return weight, bias
    
    def get_padding(self, kernel_choice):
        if kernel_choice == 3:
            padding = 1
        elif kernel_choice == 5:
            padding = 2
        else:
            padding = 3
        return padding

    def forward(self, x, weights, use_argmax=False):
        if use_argmax==True:
            #print(weights)
            weights_max_id = torch.argmax(torch.tensor(weights))
            channel_id_in = torch.div(weights_max_id, len(self.choices_kernel)*len(self.choices_channel_out), rounding_mode='trunc')
            channel_id_out = torch.div(weights_max_id, len(self.choices_kernel), rounding_mode='trunc') % len(self.choices_channel_out)

            kernel_id = weights_max_id % len(self.choices_kernel)
            #print("selected channel_in: ", self.choices_channel_in[channel_id_in])
            #print("selected channel_out: ", self.choices_channel_out[channel_id_out])
            #print("selected kernel: ", self.choices_kernel[kernel_id])
            start = (self.max_kernel - self.choices_kernel[kernel_id])//2
            weight, bias = self.get_weights_and_bias(self.choices_channel_in[channel_id_in],self.choices_channel_out[channel_id_out],self.choices_kernel[kernel_id],start)
            # pad weights and bias to max channel choice and max kernel choice
            #print(weight)
            #print(bias)
            weight  = weights[weights_max_id]*weight #torch.nn.functional.pad(weight, (start,start,start,start,0,self.max_channel_in-self.choices_channel_in[channel_id_in],0,self.max_channel_out-self.choices_channel_out[channel_id_out]))
            bias = weights[weights_max_id]*bias #torch.nn.functional.pad(bias, (0,self.max_channel_out-self.choices_channel_out[channel_id_out]))
            #print(weight)
            #print(bias)
            padding = self.get_padding(self.choices_kernel[kernel_id])
            x = torch.nn.functional.conv2d(x, weight, bias, padding=padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        else:
            weight_mix = 0
            bias_mix = 0
            arch_param_id = 0
            for i in range(len(self.choices_channel_in)):
                for j in range(len(self.choices_channel_out)):
                    for k in range(len(self.choices_kernel)):
                        start = (self.max_kernel - self.choices_kernel[k])//2
                        weight, bias = self.get_weights_and_bias(self.choices_channel_in[i],self.choices_channel_out[j],self.choices_kernel[k],start)
                        # pad weights and bias to max channel choice and max kernel choice
                        weight  = torch.nn.functional.pad(weight, (start,start,start,start,0,self.max_channel_in-self.choices_channel_in[i],0,self.max_channel_out-self.choices_channel_out[j]))
                        bias = torch.nn.functional.pad(bias, (0,self.max_channel_out-self.choices_channel_out[j]))
                        weight_mix += weights[arch_param_id]*weight
                        bias_mix += weights[arch_param_id]*bias
                        arch_param_id += 1
            #print(weight_mix)
            #print(bias_mix)
            x = torch.nn.functional.conv2d(x, weight_mix, bias_mix, padding=self.op.padding, groups=self.op.groups, dilation=self.op.dilation, stride=self.op.stride)
        return x
        
# test conv mixture
'''conv = torch.nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, bias=True)
conv.weight.data = torch.ones_like(conv.weight.data)
conv.bias.data = torch.ones_like(conv.bias.data)
conv_mix = ConvMixtureFixedInCh(conv, [1,2,3], [3,5,7])
x = torch.ones(1,3,8,8)
weights_channels = torch.tensor([0.7,0.2,0.1])
weights_kernels = torch.tensor([0.2,0.7,0.1])
# compute cross prod of weights
weights = torch.zeros(len(weights_channels)*len(weights_kernels))
for i in range(len(weights_channels)):
    for j in range(len(weights_kernels)):
        weights[i*len(weights_kernels)+j] = weights_channels[i]*weights_kernels[j]
print(weights)
print(torch.sum(weights))
out = conv(x)
print(out.shape)
out = conv_mix(x,weights)
print(out.shape)
out = conv_mix(x,weights,use_argmax=True)
print(out.shape)'''
'''conv = torch.nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, bias=True)
conv.weight.data = torch.ones_like(conv.weight.data)
conv.bias.data = torch.ones_like(conv.bias.data)
conv_mix = ConvMixtureFixedOutCh(conv,[1,2,3])
x = torch.ones(1,3,8,8)
weights_channels = torch.tensor([0.7,0.2,0.1])
out = conv_mix(x,weights_channels, use_argmax=True)'''
'''conv = torch.nn.Conv2d(3, 7, kernel_size=7, stride=1, padding=3, bias=True)
conv.weight.data = torch.ones_like(conv.weight.data)
conv.bias.data = torch.ones_like(conv.bias.data)
conv_mix = ConvMixture(conv,[1,2,3],[3,5,7],[3,5,7])
x = torch.ones(1,3,8,8)
weights_channels = torch.tensor([0.7,0.2,0.1]) # 0 -> 1
weights_kernels = torch.tensor([0.2,0.7,0.1]) # 1 -> 5
weights_channels_2 = torch.tensor([0.7,0.2,0.1]) # 2 -> 7
# compute cross prod of weights
weights = torch.zeros(len(weights_channels)*len(weights_kernels)*len(weights_channels_2))
for i in range(len(weights_channels)):
    for k in range(len(weights_channels_2)):
        for j in range(len(weights_kernels)):
            weights[i*len(weights_kernels)*len(weights_channels_2)+k*len(weights_kernels)+j] = weights_channels[i]*weights_kernels[j]*weights_channels_2[k]
print(weights)
print(sum(weights))
out = conv_mix(x,weights, use_argmax=True)
print(out.shape)

out = conv_mix(x,weights)
print(out.shape)'''

