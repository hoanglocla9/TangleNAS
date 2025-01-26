import torch.nn as nn
import torch
import torch.nn.functional as F
from optimizers.optim_factory import get_sampler, get_mixop


class MixedLinear_QO(nn.Module):
    def __init__(self, hidden_size_list, num_heads_list, linear_layer, reverse=False):
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.num_heads_list = num_heads_list
        self.register_parameter('weight', linear_layer.weight)
        self.register_parameter('bias', linear_layer.bias)

        self.max_in_dim = max(self.hidden_size_list)
        self.max_out_dim = min(self.num_heads_list) * max(self.hidden_size_list) // min(self.num_heads_list)
        if reverse:
            self.max_in_dim = min(self.num_heads_list) * max(self.hidden_size_list) // min(self.num_heads_list)
            self.max_out_dim = max(self.hidden_size_list)
        self.reverse = reverse


    def sample_weights_and_bias(self, input_dim, output_dim):
        weight = self.weight[:output_dim, :input_dim]
        if self.bias is None:
            bias = None
        else:
            bias = self.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        #print("linear layer weight", self.linear_layer.weight.shape)
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            hidden_size_argmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.hidden_size_list) if len(self.num_heads_list) > 1 else 0
            head_dim = self.hidden_size_list[hidden_size_argmax_id] // self.num_heads_list[num_heads_argmax_id] 
            
            
            if not self.reverse:
                weight, bias = self.sample_weights_and_bias(
                    self.hidden_size_list[hidden_size_argmax_id], self.num_heads_list[num_heads_argmax_id] * head_dim)
            else:
                weight, bias = self.sample_weights_and_bias(self.num_heads_list[num_heads_argmax_id] * head_dim, 
                    self.hidden_size_list[hidden_size_argmax_id])
                
            weight = weights[weights_max]*weight
            if bias is not None:
                bias = weights[weights_max]*bias
            else:
                bias = None

            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            m = 0
            for i in range(len(self.hidden_size_list)):
                for j in range(len(self.num_heads_list)):
                    head_dim = self.hidden_size_list[i] // self.num_heads_list[j] 
                    if self.reverse:
                        weight, bias = self.sample_weights_and_bias(
                            self.num_heads_list[j] * head_dim, self.hidden_size_list[i])

                    else:
                        weight, bias = self.sample_weights_and_bias(
                            self.hidden_size_list[i], self.num_heads_list[j] * head_dim)
                    weight = F.pad(
                        weight, (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
                    if bias is not None:
                        bias = F.pad(bias, (0, self.max_out_dim -
                            bias.shape[-1]), "constant", 0)
                    weights_mix += weights[m]*weight
                    
                    if bias is not None:
                        bias_mix += weights[m]*bias
                    else:
                        bias_mix = None
                    m += 1
            out = F.linear(x, weights_mix, bias_mix)

        return out
    

class MixedLinear_KV(nn.Module):
    def __init__(self, hidden_size_list, num_heads_list, num_heads_kv, linear_layer):
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.num_heads_kv = num_heads_kv

        self.register_parameter('weight', linear_layer.weight)
        self.register_parameter('bias', linear_layer.bias)
        
        self.max_in_dim = max(self.hidden_size_list)
        max_head_dim = max(self.hidden_size_list) // min(num_heads_list)
        self.max_out_dim = num_heads_kv * max_head_dim
        self.num_heads_list = num_heads_list

    def sample_weights_and_bias(self, input_dim, output_dim):
        weight = self.weight[:output_dim, :input_dim]
        if self.bias is None:
            bias = None
        else:
            bias = self.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        #print("linear layer weight", self.linear_layer.weight.shape)
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            hidden_size_argmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.hidden_size_list) if len(self.num_heads_list) > 1 else 0
            head_dim = self.hidden_size_list[hidden_size_argmax_id] // self.num_heads_list[num_heads_argmax_id]
            weight, bias = self.sample_weights_and_bias(
                    self.hidden_size_list[hidden_size_argmax_id], self.num_heads_kv * head_dim)
            weight = weights[weights_max]*weight

            if bias is not None:
                bias = weights[weights_max]*bias
            else:
                bias = None
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            m = 0
            for i in range(len(self.hidden_size_list)):
                for j in range(len(self.num_heads_list)):
                    weight, bias = self.sample_weights_and_bias(
                            self.hidden_size_list[i], self.num_heads_kv * self.hidden_size_list[i] // self.num_heads_list[j])
                    weight = F.pad(
                        weight, (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
                    if bias is not None:
                        bias = F.pad(bias, (0, self.max_out_dim -
                            bias.shape[-1]), "constant", 0)
                    weights_mix += weights[m]*weight
                        
                    if bias is not None:
                        bias_mix += weights[m]*bias
                    else:
                        bias_mix = None
                    m += 1
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

