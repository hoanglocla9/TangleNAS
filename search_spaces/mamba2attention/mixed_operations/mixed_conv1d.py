import torch.nn as nn
import torch

class MixedConv1dV2(nn.Module):
    def __init__(self, n_channels_list, conv1d) -> None: 
        ## assume that in_channels = out_channels
        ## Not support for multiple choices of kernel_size yet!
        super().__init__()
        self.n_channels_list = n_channels_list
        self.max_n_channels = max(n_channels_list)
        self.conv1d = conv1d

    def sample_weights_and_bias(self, n_channels):
        weight = self.conv1d.weight[:n_channels,:,:]
        if self.conv1d.bias is None:
            bias = None
        else:
            bias = self.conv1d.bias[:n_channels]
        return weight, bias
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight, bias = self.sample_weights_and_bias(self.n_channels_list[weights_max])
            # pad weights and bias
            weight = weights[weights_max]*weight
            if bias is not None:
                bias =  weights[weights_max]*bias
            out =  torch.nn.functional.conv1d(x[:,:self.n_channels_list[weights_max],:], weight=weight, bias=bias, 
                    stride=self.conv1d.stride, padding=self.conv1d.padding, dilation=self.conv1d.dilation, 
                    groups=self.n_channels_list[weights_max])
            return out
        else:
            weights_mix = 0
            bias_mix = 0
            for i, n_channels in enumerate(self.n_channels_list):
                weight, bias = self.sample_weights_and_bias(n_channels)
                # pad weights and bias
                weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, self.max_n_channels - weight.size(0)), "constant", 0)
                if bias is None:
                    bias = None
                else:
                    bias = torch.nn.functional.pad(bias, (0, self.max_n_channels - bias.shape[0]), "constant", 0)
                weights_mix += weights[i]*weight
                if bias is not None:
                    bias_mix += weights[i]*bias
                else:
                    bias_mix = None
            out =  torch.nn.functional.conv1d(x, weight=weights_mix, bias=bias_mix, 
                    stride=self.conv1d.stride, padding=self.conv1d.padding, dilation=self.conv1d.dilation, 
                    groups=self.max_n_channels)
            return out


class MixedConv1dV2_MHA(nn.Module):
    def __init__(self, embed_dim_list, num_heads_list, conv1d) -> None: 
        ## assume that in_channels = out_channels
        ## Not support for multiple choices of kernel_size yet!
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.num_heads_list = num_heads_list

        self.max_n_channels = max(self.embed_dim_list)//min(self.num_heads_list) * 3 * (max(self.num_heads_list))

        self.conv1d = conv1d

    def sample_weights_and_bias(self, n_channels):
        weight = self.conv1d.weight[:n_channels,:,:]
        if self.conv1d.bias is None:
            bias = None
        else:
            bias = self.conv1d.bias[:n_channels]
        return weight, bias
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            embed_dim_argxmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.embed_dim_list)
            n_channels = self.embed_dim_list[embed_dim_argxmax_id] // self.num_heads_list[num_heads_argmax_id] * 3 * self.num_heads_list[num_heads_argmax_id]
            weight, bias = self.sample_weights_and_bias(n_channels)
            # pad weights and bias
            weight = weights[weights_max]*weight
            if bias is not None:
                bias =  weights[weights_max]*bias
            out =  torch.nn.functional.conv1d(x[:,:n_channels,:], weight=weight, bias=bias, 
                    stride=self.conv1d.stride, padding=self.conv1d.padding, dilation=self.conv1d.dilation, 
                    groups=n_channels)
            return out
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.embed_dim_list)):
                for j in range(len(self.num_heads_list)):
                    n_channels = self.embed_dim_list[i] // self.num_heads_list[j] * 3 * self.num_heads_list[j]
                    weight, bias = self.sample_weights_and_bias(n_channels)
                    # pad weights and bias
                    weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, self.max_n_channels - weight.size(0)), "constant", 0)
                    if bias is None:
                        bias = None
                    else:
                        bias = torch.nn.functional.pad(bias, (0, self.max_n_channels - bias.shape[0]), "constant", 0)
                    weights_mix += weights[k]*weight
                    if bias is not None:
                        bias_mix += weights[k]*bias
                    else:
                        bias_mix = None
                    k += 1

            out =  torch.nn.functional.conv1d(x, weight=weights_mix, bias=bias_mix, 
                    stride=self.conv1d.stride, padding=self.conv1d.padding, dilation=self.conv1d.dilation, 
                    groups=self.max_n_channels)
            return out


# test mixed layer norm
# if __name__ == "__main__":
#     conv1d_layer = torch.nn.Conv1d(1024, 1024, bias=True, kernel_size=4,groups=1024,padding=3)
#     conv1d_layer.weight.data = torch.ones(1024, 1, 67)
#     conv1d_layer.bias.data = torch.ones(1024)

#     channels_list = [512, 768, 1024]
#     max_channels = 1024
#     mixed_conv1d = MixedConv1dV2(channels_list, conv1d_layer)
#     x = torch.ones(1, 1024, 64)
#     weights = [0, 0, 1.0]
#     out = mixed_conv1d(x, weights)
#     print(out.shape)
#     print(out)
#     out = mixed_conv1d(x, weights, use_argmax=True)
#     print(out.shape)
#     print(out)