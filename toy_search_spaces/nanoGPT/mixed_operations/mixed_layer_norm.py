import torch.nn as nn
import torch
import torch.nn.functional as F

class MixedLayerNormV2(nn.Module):
    def __init__(self, embed_dim_list, max_embed_dim, layer_norm) -> None:
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.max_embed_dim = max_embed_dim
        self.layer_norm = layer_norm

    def sample_weights_and_bias(self, emb_dim):
        weight = self.layer_norm.weight[:emb_dim]
        if self.layer_norm.bias is None:
            bias = None
        else:
            bias = self.layer_norm.bias[:emb_dim]
        return weight, bias
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight, bias = self.sample_weights_and_bias(self.embed_dim_list[weights_max])
            # pad weights and bias
            weight = weights[weights_max]*weight
            if bias is not None:
                bias =  weights[weights_max]*bias
            out =  torch.nn.functional.layer_norm(x[:,:,:self.embed_dim_list[weights_max]], weight.shape, weight=weight, bias=bias, eps=self.layer_norm.eps)
            # pad out
            return out
        else:
            weights_mix = 0
            bias_mix = 0
            for i, embed_dim in enumerate(self.embed_dim_list):
                weight, bias = self.sample_weights_and_bias(embed_dim)
                # pad weights and bias
                weight = torch.nn.functional.pad(weights[i]*weight, (0,self.max_embed_dim - weight.shape[-1]), "constant", 0)
                if bias is None:
                    bias = None
                else:
                    bias = torch.nn.functional.pad(weights[i]*bias, (0,self.max_embed_dim - bias.shape[-1]), "constant", 0)
                weights_mix += weight
                if bias is not None:
                    bias_mix += bias
                else:
                    bias_mix = None
            #print(weights_mix, bias_mix)
            out =  torch.nn.functional.layer_norm(x, weights_mix.shape, weight=weights_mix, bias=bias_mix, eps=self.layer_norm.eps)
            return out

# test mixed layer norm
'''if __name__ == "__main__":
    layer_norm = nn.LayerNorm(10)
    layer_norm.weight.data = torch.ones(10)
    layer_norm.bias.data = torch.ones(10)
    embed_dim_list = [6, 8, 10]
    max_embed_dim = 10
    mixed_layer_norm = MixedLayerNormV2(embed_dim_list, max_embed_dim, layer_norm)
    x = torch.ones(1, 2, 10)
    weights = [0.1, 0.7, 0.2]
    out = mixed_layer_norm(x, weights)
    print(out.shape)
    print(out)
    out = mixed_layer_norm(x, weights, use_argmax=True)
    print(out.shape)
    print(out)'''