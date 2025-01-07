import torch.nn as nn
import torch

class MixedRMSNormGatedV2(nn.Module):
    def __init__(self, hidden_size_list, max_hidden_size, rms_norm) -> None:
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.max_hidden_size = max_hidden_size
        self.rms_norm = rms_norm

    def sample_weights(self, emb_dim):
        weight = self.rms_norm.weight[:emb_dim]
        return weight
    
    def forward(self, x, weights, use_argmax=False, gate=None):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if gate is not None:
            x = x * nn.functional.silu(gate.to(torch.float32))

        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight = self.sample_weights(self.hidden_size_list[weights_max])
            # pad weights and bias
            weight = weights[weights_max]*weight
            #out =  torch.nn.functional.layer_norm(x[:,:,:self.embed_dim_list[weights_max]], weight.shape, weight=weight, bias=bias, eps=self.layer_norm.eps)
            variance = x[:,:,:self.hidden_size_list[weights_max]].pow(2).mean(-1, keepdim=True)
            tmp_x = x[:,:,:self.hidden_size_list[weights_max]] * torch.rsqrt(variance + self.rms_norm.variance_epsilon)
            out = weight * tmp_x
            # pad out
            return out
        else:
            weights_mix = 0
            bias_mix = 0
            for i, embed_dim in enumerate(self.hidden_size_list):
                weight = self.sample_weights(embed_dim)
                # pad weights and bias
                weight = torch.nn.functional.pad(weight, (0,self.max_hidden_size - weight.shape[-1]), "constant", 0)
                
                weights_mix += weights[i]*weight
            
            variance = x.pow(2).mean(-1, keepdim=True)
            tmp_x = x * torch.rsqrt(variance + self.rms_norm.variance_epsilon)
            out = weights_mix * tmp_x

            return out

# test mixed layer norm
'''
if __name__ == "__main__":
    rms_norm = Mamba2RMSNorm(1024)
    # rms_norm.weight.data = torch.ones(1024)
    hidden_size_list = [512, 768, 1024]
    max_hidden_size = 1024
    mixed_rms_norm = MixedRMSNormV2(hidden_size_list, max_hidden_size, rms_norm)
    x = torch.ones(1, 2, 1024)
    weights = [0.1, 0.7, 0.2]
    out = mixed_rms_norm(x, weights)
    print(out.shape)
    print(out)
    out = mixed_rms_norm(x, weights, use_argmax=True)
    print(out.shape)
    print(out)
    print(rms_norm(x).shape)
'''