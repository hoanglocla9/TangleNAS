import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedLinearV2Emb(nn.Module):
    def __init__(self, emb_dim_list, max_out_dim, linear_layer):
        super().__init__()
        self.emb_dim_list = emb_dim_list
        self.linear_layer = linear_layer
        self.max_out_dim = max_out_dim

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        if linear_layer.bias is not None:
            bias = linear_layer.bias[:output_dim]
        else:
            bias = None
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight, bias = self.sample_weights_and_bias(
                self.emb_dim_list[weights_max], self.emb_dim_list[weights_max], self.linear_layer)
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
            # print("X shape: ", x.shape)
            for i in range(len(self.emb_dim_list)):
                weight, bias = self.sample_weights_and_bias(
                self.emb_dim_list[i], self.emb_dim_list[i], self.linear_layer)
                # pad weights and bias
                weight = F.pad(weights[i]*weight, (0, self.max_out_dim -
                           weight.shape[-1], 0, self.max_out_dim - weight.shape[-1]), "constant", 0)
                if bias is not None:
                    bias = F.pad(weights[i]*bias, (0, self.max_out_dim -
                         bias.shape[-1]), "constant", 0)
                else:
                    bias = None
                weights_mix += weight
                if bias is not None:
                    bias_mix += bias
                else:
                    bias_mix = None
            #print("weights_mix:", weights_mix)
            #print("bias_mix:", bias_mix)
            out = F.linear(x, weights_mix, bias_mix)

        return out
    

# test mixed linear emb
'''if __name__ == "__main__":
    emb_dim_list = [6, 8, 10]
    max_out_dim = max(emb_dim_list)
    linear_layer = nn.Linear(max_out_dim, max_out_dim, bias=True)
    linear_layer.weight.data = torch.ones_like(linear_layer.weight.data)
    linear_layer.bias.data = torch.ones_like(linear_layer.bias.data)
    mixed_linear_emb = MixedLinearV2Emb(emb_dim_list, max_out_dim, linear_layer)
    x = torch.ones(1,10)
    weights = [0.1, 0.7, 0.2]
    weight_true = 0
    bias_true = 0
    for i in range(len(emb_dim_list)):
        weight_true += torch.nn.functional.pad(weights[i]*linear_layer.weight[:emb_dim_list[i], :emb_dim_list[i]], (0, max_out_dim -
                           emb_dim_list[i], 0, max_out_dim - emb_dim_list[i]), "constant", 0)
        bias_true += torch.nn.functional.pad(weights[i]*linear_layer.bias[:emb_dim_list[i]], (0, max_out_dim -
                           emb_dim_list[i]), "constant", 0)
    out_true = F.linear(x, weight_true, bias_true)
    out = mixed_linear_emb(x, weights)
    print(out)
    assert torch.allclose(out, out_true)
    out = mixed_linear_emb(x, weights, use_argmax=True)
    print(out)'''

class MixedLinear(nn.Module):
    def __init__(self, input_dim_list, linear_layer, reverse=False):
        super().__init__()
        self.input_dim_list = input_dim_list
        self.linear_layer = linear_layer
        self.max_in_dim = max(self.input_dim_list)
        self.reverse = reverse

    def sample_weights_and_bias(self, dim, linear_layer):
        if not self.reverse:
            weight = linear_layer.weight[:, :dim]
            bias = linear_layer.bias
        else:
            weight = linear_layer.weight[:dim, :]
            if linear_layer.bias is None:
                bias = None
            else:
                bias = linear_layer.bias[:dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight, bias = self.sample_weights_and_bias(
                    self.input_dim_list[weights_max], self.linear_layer)
            if not self.reverse:
                weight = weights[weights_max]*weight
                
            else:
                weight = weights[weights_max]*weight
                if bias is not None:
                    bias = weights[weights_max]*bias
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.input_dim_list)):
                weight, bias = self.sample_weights_and_bias(
                        self.input_dim_list[i], self.linear_layer)
                if not self.reverse:
                    weight = F.pad(weight, (0, self.max_in_dim-weight.shape[-1]), "constant", 0)
                else:
                    weight = F.pad(weight, (0,0,0, self.max_in_dim-weight.shape[-2]), "constant", 0)
                    if bias is not None:
                        bias = F.pad(bias, (0, self.max_in_dim -
                             bias.shape[-1]), "constant", 0)
                    #print(weight.shape)
                weights_mix += weights[k]*weight
                if bias is not None:
                    if not self.reverse:
                        bias_mix += weights[k]*bias
                    else:
                        bias_mix = bias
                else:
                    bias_mix = None
                k = k+1
            out = F.linear(x, weights_mix, bias_mix)

        return out
    

