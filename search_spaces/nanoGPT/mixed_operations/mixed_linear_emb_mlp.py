import torch.nn as nn
import torch
import torch.nn.functional as F
from optimizers.optim_factory import get_sampler, get_mixop
class MixedLinearV2(nn.Module):
    def __init__(self, embed_dim_list,  mlp_ratio_list, linear_layer, reverse=False):
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.mlp_ratio_list = mlp_ratio_list
        self.linear_layer = linear_layer
        self.max_in_dim = max(self.embed_dim_list)
        self.max_out_dim = max(self.embed_dim_list)*max(self.mlp_ratio_list)
        self.reverse = reverse
        if reverse:
            self.max_out_dim = max(self.embed_dim_list)
            self.max_in_dim = max(self.embed_dim_list) * \
                max(self.mlp_ratio_list)

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        if linear_layer.bias is None:
            bias = None
        else:
            bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        #print("linear layer weight", self.linear_layer.weight.shape)
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            #print("weights_max", weights_max)
            embed_dim_argmax_id = torch.div(weights_max, len(self.mlp_ratio_list), rounding_mode='floor')
            mlp_ratio_argmax_id = weights_max%len(self.embed_dim_list)
            #print("Selected emb", self.embed_dim_list[embed_dim_argmax_id])
            #print("Selected mlp", self.mlp_ratio_list[mlp_ratio_argmax_id])
            if self.reverse:
                weight, bias = self.sample_weights_and_bias(
                    self.mlp_ratio_list[mlp_ratio_argmax_id]*self.embed_dim_list[embed_dim_argmax_id], self.embed_dim_list[embed_dim_argmax_id], self.linear_layer)
            else:
                weight, bias = self.sample_weights_and_bias(
                    self.embed_dim_list[embed_dim_argmax_id], self.mlp_ratio_list[mlp_ratio_argmax_id]*self.embed_dim_list[embed_dim_argmax_id], self.linear_layer)
            # pad weights and bias
            #print(weight)
            #print(weight.shape)
            #print(bias)
            #print(bias.shape)
            weight = weights[weights_max]*weight
            if bias is not None:
                bias = weights[weights_max]*bias
            else:
                bias = None
            #print(weight)
            #print(bias)
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.embed_dim_list)):
                for j in range(len(self.mlp_ratio_list)):
                    if self.reverse:
                        weight, bias = self.sample_weights_and_bias(
                        self.mlp_ratio_list[j]*self.embed_dim_list[i], self.embed_dim_list[i], self.linear_layer)
                    else:
                        weight, bias = self.sample_weights_and_bias(
                        self.embed_dim_list[i], self.embed_dim_list[i]*self.mlp_ratio_list[j], self.linear_layer)
                    # pad weights and bias
                    #print(weight.shape)
                    #print("Choice emb",self.embed_dim_list[i])
                    #print("Choice mlp",self.mlp_ratio_list[j])
                    #print(weights[k]*weight)
                    #print(weight.shape)
                    #print(weights[k]*bias)
                    #print(bias.shape)
                    weight = F.pad(
                    weight, (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
                    #print(weight.shape)
                    if bias is not None:
                        bias = F.pad(bias, (0, self.max_out_dim -
                             bias.shape[-1]), "constant", 0)
                    weights_mix += weights[k]*weight
                    
                    if bias is not None:
                        bias_mix += weights[k]*bias
                    else:
                        bias_mix = None
                    k = k+1
            #print("weights_mix", weights_mix)
            #print("bias_mix", bias_mix)
            out = F.linear(x, weights_mix, bias_mix)

        return out
    
# test mixed linear
'''if __name__ == "__main__":
    input_dim_list = [6,8,10]
    output_dim_list = [1,2,3]
    mixop  = get_mixop("darts_v1")
    sampler = get_sampler("darts_v1")
    max_out_dim = max(input_dim_list)*max(output_dim_list)
    max_in_dim = max(input_dim_list)
    linear_layer = nn.Linear(max_in_dim, max_out_dim, bias=True)
    linear_layer.weight.data = torch.ones(max_out_dim, max_in_dim)
    linear_layer.bias.data = torch.ones(max_out_dim)
    mixed_linear = MixedLinearV2(input_dim_list, output_dim_list, linear_layer)
    x = torch.ones(1, max_in_dim)
    weights1 = torch.nn.Parameter(1e-3*torch.randn([len(input_dim_list)]))
    weights2 = torch.nn.Parameter(1e-3*torch.randn([len(output_dim_list)]))
    # take cross product of tensors
    weights = torch.einsum('i,j->ij', weights1, weights2)
    weights = weights.reshape(-1)
    print(weights)
    print("Sum of weights: ", torch.sum(weights))
    #out = mixed_linear(x, weights)
    #print(out)
    out = mixed_linear(x, weights)
    linear_layer_2 = nn.Linear(max_out_dim, max_in_dim, bias=True)
    linear_layer_2.weight.data = torch.ones(max_in_dim, max_out_dim)
    linear_layer_2.bias.data = torch.ones(max_in_dim)
    mixed_linear = MixedLinearV2(input_dim_list, output_dim_list, linear_layer_2, reverse=True)
    x = torch.ones(1, max_out_dim)
    weights1 = torch.nn.Parameter(torch.randn([len(input_dim_list)]))
    weights2 = torch.nn.Parameter(torch.randn([len(output_dim_list)]))
    # cross product of tensors
    weights_1_softmax = F.softmax(weights1, dim=0)
    weights_2_softmax = F.softmax(weights2, dim=0)
    weights_old = torch.zeros([len(input_dim_list)*len(output_dim_list)])
    for i in range(len(input_dim_list)):
        for j in range(len(output_dim_list)):
            weights_old[i*len(output_dim_list)+j] = weights_1_softmax[i]*weights_2_softmax[j]
    weights_old = weights_old.reshape(-1)
    print(weights_old)
    # take cross sum of tensors
    weight_final = torch.zeros([len(input_dim_list)*len(output_dim_list)])
    for i in range(len(input_dim_list)):
        for j in range(len(output_dim_list)):
            weight_final[i*len(output_dim_list)+j] = weights1[i]+weights2[j]
    weights = weight_final
    # sample weights
    weights = sampler.sample_step([weights])
    print(weights1)
    print(weights2)
    print(torch.nn.functional.softmax(weights[0],dim=-1))
    print(weights)
    print("Sum of weights: ", torch.sum(weights))
    #out = mixed_linear(x, weights)
    #print(out)
    out = mixed_linear(x, weights, use_argmax=True)
    print(out)'''

