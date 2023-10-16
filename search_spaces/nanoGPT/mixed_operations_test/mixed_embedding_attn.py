import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedEmbeddingAttention(nn.Module):
    def __init__(self, embed_dim_list, max_embed_dim, linear_layer) -> None:
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.linear_layer = linear_layer
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, embed_dim):
        weight = self.linear_layer.weight[:embed_dim*3,:embed_dim]
        if self.linear_layer.bias is not None:
            bias = self.linear_layer.bias[:embed_dim*3]
        else:
            bias = None
        return weight, bias
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            linear_weight, linear_bias = self.sample_weights_and_bias(self.embed_dim_list[weights_max])
            linear_weight = weights[weights_max]*linear_weight
            if linear_bias is not None:
                linear_bias = weights[weights_max]*linear_bias
            else:
                linear_bias = None
            #print("linear_weight_padded:", linear_weight_padded)
            #print("linear_bias_padded:", linear_bias_padded)
            out = F.linear(x, linear_weight, linear_bias)
            return out
        else:
            linear_weight_mixture = 0
            linear_bias_mixture = 0
            i=0
            for embed_dim in self.embed_dim_list:
                linear_weight, linear_bias = self.sample_weights_and_bias(embed_dim)
                linear_weight_padded = F.pad(linear_weight, (0,self.max_embed_dim - linear_weight.shape[-1],0, 3*self.max_embed_dim - linear_weight.shape[-2]), "constant", 0)
                linear_weight_mixture += weights[i] * linear_weight_padded
                if linear_bias is not None:
                    linear_bias_padded = F.pad(linear_bias, (0,3*self.max_embed_dim - linear_bias.shape[-1]), "constant", 0)
                    linear_bias_mixture += weights[i] * linear_bias_padded
                else:
                    linear_bias_padded = None
                    linear_bias_mixture = None
                i+=1
            #print(linear_weight_mixture)
            #print(linear_bias_mixture)
            out = F.linear(x, linear_weight_mixture, linear_bias_mixture)
            return out
        

# test mixed linear emb
'''if __name__ == "__main__":
    emb_dim_list = [10, 20, 30]
    max_out_dim = max(emb_dim_list)*3
    max_embed_dim = max(emb_dim_list)
    linear_layer = nn.Linear(max_embed_dim, max_out_dim)
    linear_layer.weight.data = torch.ones_like(linear_layer.weight.data)
    linear_layer.bias.data = torch.ones_like(linear_layer.bias.data)
    mixed_linear_emb = MixedEmbeddingAttention(emb_dim_list, max_embed_dim, linear_layer)
    x = torch.ones(1, max_embed_dim)
    weights = [0.1, 0.7, 0.2]
    out = mixed_linear_emb(x, weights)
    print(out.shape)
    print(out)
    out = mixed_linear_emb(x, weights, use_argmax=True)
    print(out.shape)
    print(out)
    print(out.shape)'''
