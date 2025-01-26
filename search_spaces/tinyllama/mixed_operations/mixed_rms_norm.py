import torch.nn as nn
import torch
import torch.nn.functional as F

class MixedRMSNorm(nn.Module):
    def __init__(self, hidden_size_list, rms_norm) -> None:
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.register_parameter('weight', rms_norm.weight)
        self.variance_epsilon = rms_norm.variance_epsilon

    def sample_weights_and_bias(self, hidden_size):
        weight = self.weight[:hidden_size]
        return weight
    
    def forward(self, x, weights, use_argmax=False):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight = self.sample_weights_and_bias(self.hidden_size_list[weights_max])
            # pad weights and bias'
            weight = weights[weights_max]*weight
            return weight * x.to(input_dtype)
        
        else:
            weights_mix = 0
            for i, hidden_size in enumerate(self.hidden_size_list):
                weight = self.sample_weights_and_bias(hidden_size)
                # pad weights and bias
                weight = torch.nn.functional.pad(weight, (0,self.max_embed_dim - weight.shape[-1]), "constant", 0)
                weights_mix = weights[i] * weight
            return weights_mix * x.to(input_dtype)
