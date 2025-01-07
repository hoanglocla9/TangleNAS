import torch
import torch.nn as nn
import torch.nn.functional as F
# from optimizers.optim_factory import get_sampler, get_mixop
class MixedLinearV2_InProj(nn.Module):
    def __init__(self, hidden_size_list, num_heads_list, expand, n_groups, ssm_state_size,  linear_layer):
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.max_in_dim = max(hidden_size_list)
        self.num_heads_list = num_heads_list
        self.max_num_heads = max(num_heads_list)
        #self.projection_size_list = [2*(int(expand * hidden_size) + n_groups * self.ssm_state_size) + self.num_heads for self.hidden_size_list]
        self.max_out_dim = 2*(int(expand * self.max_in_dim) + n_groups * ssm_state_size) + self.max_num_heads
        self.expand = expand
        self.n_groups = n_groups
        self.ssm_state_size = ssm_state_size
        self.linear_layer = linear_layer

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        if linear_layer.bias is None:
            bias = None
        else:
            bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            hidden_size_argmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.hidden_size_list)
            weight, bias = self.sample_weights_and_bias(self.hidden_size_list[hidden_size_argmax_id],
                2*(int(self.expand * self.hidden_size_list[hidden_size_argmax_id]) + self.n_groups * self.ssm_state_size) + self.num_heads_list[num_heads_argmax_id], self.linear_layer)
            # pad weights and bias
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
            for i in range(len(self.hidden_size_list)):
                for j in range(len(self.num_heads_list)):
                    projection_size = 2*(int(self.expand * self.hidden_size_list[i]) + self.n_groups * self.ssm_state_size) + self.num_heads_list[j]
                    weight, bias = self.sample_weights_and_bias(
                        self.hidden_size_list[i], projection_size, self.linear_layer)
                    weight = F.pad(weight, 
                        (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
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
            out = F.linear(x, weights_mix, bias_mix)

        return out
    

class MixedLinearV2_InProj_MHA(nn.Module):
    def __init__(self, embed_dim_list, num_heads_list, mlp_dim, linear_layer):  # expand, n_groups, ssm_state_size, 
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.num_heads_list = num_heads_list
        self.mlp_dim = mlp_dim

        max_head_dim =  max(self.embed_dim_list)//min(self.num_heads_list)
        self.max_d_in = max(self.embed_dim_list)
        self.max_d_out = max_head_dim * 3 * (max(self.num_heads_list))
        
        self.linear_layer = linear_layer

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        if linear_layer.bias is None:
            bias = None
        else:
            bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            embed_dim_argxmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.embed_dim_list)
            weight, bias = self.sample_weights_and_bias(self.embed_dim_list[embed_dim_argxmax_id],
                self.embed_dim_list[embed_dim_argxmax_id]//self.num_heads_list[num_heads_argmax_id] * 3 * self.num_heads_list[num_heads_argmax_id], self.linear_layer)
            # pad weights and bias
            weight = weights[weights_max]*weight
            if bias is not None:
                bias = weights[weights_max]*bias
            else:
                bias = None
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.embed_dim_list)):
                for j in range(len(self.num_heads_list)):
                    projection_size = self.embed_dim_list[i]//self.num_heads_list[j] * 3 * self.num_heads_list[j]
                    weight, bias = self.sample_weights_and_bias(
                        self.embed_dim_list[i], projection_size, self.linear_layer)
                    weight = F.pad(weight, 
                        (0, self.max_d_in-weight.shape[-1], 0, self.max_d_out - weight.shape[-2]), "constant", 0)
                    if bias is not None:
                        bias = F.pad(bias, (0, self.max_d_out -
                             bias.shape[-1]), "constant", 0)
                    weights_mix += weights[k]*weight
                    if bias is not None:
                        bias_mix += weights[k]*bias
                    else:
                        bias_mix = None
                    k = k+1
            out = F.linear(x, weights_mix, bias_mix)

        return out
    

class MixedLinearV2_OutProj(nn.Module):
    def __init__(self, hidden_size_list, expand, linear_layer):
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.max_in_dim = int(expand * max(hidden_size_list))
        #self.projection_size_list = [2*(int(expand * hidden_size) + n_groups * self.ssm_state_size) + self.num_heads for self.hidden_size_list]
        self.max_out_dim = max(hidden_size_list)
        self.expand = expand
        self.linear_layer = linear_layer

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        if linear_layer.bias is None:
            bias = None
        else:
            bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            hidden_size_argmax_id = torch.argmax(torch.tensor(weights), dim=-1)
            weight, bias = self.sample_weights_and_bias(int(self.expand * self.hidden_size_list[hidden_size_argmax_id]),
                self.hidden_size_list[hidden_size_argmax_id], self.linear_layer)
            # pad weights and bias
            weight = weights[hidden_size_argmax_id]*weight
            if bias is not None:
                bias = weights[hidden_size_argmax_id]*bias
            else:
                bias = None
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.hidden_size_list)):
                weight, bias = self.sample_weights_and_bias(
                        int(self.expand * self.hidden_size_list[i]), self.hidden_size_list[i], self.linear_layer)
                weight = F.pad(weight, (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
                weights_mix += weights[i]*weight
                if bias is not None:
                    bias_mix += weights[i]*bias
                else:
                    bias_mix = None
            out = F.linear(x, weights_mix, bias_mix)
        return out
    

class MixedLinearV2_OutProj_MHA(nn.Module):
    def __init__(self, embed_dim_list, num_heads_list, mlp_dim, linear_layer):  # expand, n_groups, ssm_state_size, 
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.num_heads_list = num_heads_list
        self.mlp_dim = mlp_dim

        max_head_dim =  max(self.embed_dim_list)//min(self.num_heads_list)
        self.max_d_in = max_head_dim * max(self.num_heads_list) + self.mlp_dim // 2

        self.max_d_out = max(self.embed_dim_list)
        self.linear_layer = linear_layer

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        if linear_layer.bias is None:
            bias = None
        else:
            bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            embed_dim_argxmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.embed_dim_list)
            weight, bias = self.sample_weights_and_bias(
                self.embed_dim_list[embed_dim_argxmax_id]//self.num_heads_list[num_heads_argmax_id] * self.num_heads_list[num_heads_argmax_id] + self.mlp_dim //2,
                self.embed_dim_list[embed_dim_argxmax_id], self.linear_layer)
            # pad weights and bias
            weight = weights[weights_max]*weight
            if bias is not None:
                bias = weights[weights_max]*bias
            else:
                bias = None
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.embed_dim_list)):
                for j in range(len(self.num_heads_list)):
                    projection_size = self.embed_dim_list[i]//self.num_heads_list[j] * self.num_heads_list[i] + self.mlp_dim //2
                    weight, bias = self.sample_weights_and_bias(
                        projection_size, self.embed_dim_list[i], self.linear_layer)
                    weight = F.pad(weight, 
                        (0, self.max_d_in-weight.shape[-1], 0, self.max_d_out - weight.shape[-2]), "constant", 0)
                    if bias is not None:
                        bias = F.pad(bias, (0, self.max_d_out -
                             bias.shape[-1]), "constant", 0)
                    weights_mix += weights[k]*weight
                    if bias is not None:
                        bias_mix += weights[k]*bias
                    else:
                        bias_mix = None
                    k = k+1
            out = F.linear(x, weights_mix, bias_mix)

        return out
    
# # test mixed linear emb
# if __name__ == "__main__":
#     num_heads_list = [16, 24, 32]
#     hidden_size_list = [384, 768, 1024] 
#     expand = 2
#     input_dim_list = hidden_size_list
#     output_dim_list = num_heads_list
#     max_in_dim = max(input_dim_list)
#     max_out_dim = 2*(int(expand * max_in_dim) + 1 * 128) + max(num_heads_list)

#     # mixop  = get_mixop("darts_v1")
#     # sampler = get_sampler("darts_v1")

#     linear_layer = nn.Linear(max_in_dim, max_out_dim, bias=True)
#     linear_layer.weight.data = torch.ones(max_out_dim, max_in_dim)
#     linear_layer.bias.data = torch.ones(max_out_dim)
#     mixed_linear = MixedLinearV2(input_dim_list, output_dim_list, 2, 1, 128, linear_layer)

#     x = torch.ones(1, max_in_dim)
#     weights1 = torch.nn.Parameter(1e-3*torch.randn([len(input_dim_list)]))
#     weights2 = torch.nn.Parameter(1e-3*torch.randn([len(output_dim_list)]))
#     # take cross product of tensors
#     weights = torch.einsum('i,j->ij', weights1, weights2)
#     weights = weights.reshape(-1)
#     print(weights)
#     print("Sum of weights: ", torch.sum(weights))
#     out = mixed_linear(x, weights)
#     print(out)
#     # out = mixed_linear(x, weights)
#     # linear_layer_2 = nn.Linear(max_out_dim, max_in_dim, bias=True)
#     # linear_layer_2.weight.data = torch.ones(max_in_dim, max_out_dim)
#     # linear_layer_2.bias.data = torch.ones(max_in_dim)
#     # mixed_linear = MixedLinearV2(input_dim_list, output_dim_list, linear_layer_2, reverse=True)
#     # x = torch.ones(1, max_out_dim)
#     # weights1 = torch.nn.Parameter(torch.randn([len(input_dim_list)]))
#     # weights2 = torch.nn.Parameter(torch.randn([len(output_dim_list)]))
#     # # cross product of tensors
#     # weights_1_softmax = F.softmax(weights1, dim=0)
#     # weights_2_softmax = F.softmax(weights2, dim=0)
#     # weights_old = torch.zeros([len(input_dim_list)*len(output_dim_list)])
#     # for i in range(len(input_dim_list)):
#     #     for j in range(len(output_dim_list)):
#     #         weights_old[i*len(output_dim_list)+j] = weights_1_softmax[i]*weights_2_softmax[j]
#     # weights_old = weights_old.reshape(-1)
#     # print(weights_old)
#     # take cross sum of tensors
#     # weight_final = torch.zeros([len(input_dim_list)*len(output_dim_list)])
#     # for i in range(len(input_dim_list)):
#     #     for j in range(len(output_dim_list)):
#     #         weight_final[i*len(output_dim_list)+j] = weights1[i]+weights2[j]
#     # weights = weight_final
#     # # sample weights
#     # weights = sampler.sample_step([weights])
#     # print(weights1)
#     # print(weights2)
#     # print(torch.nn.functional.softmax(weights[0],dim=-1))
#     # print(weights)
#     # print("Sum of weights: ", torch.sum(weights))
#     # #out = mixed_linear(x, weights)
#     # #print(out)
#     # out = mixed_linear(x, weights, use_argmax=True)
#     # print(out)



