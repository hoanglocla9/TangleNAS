import torch.nn as nn
import torch
import torch.nn.functional as F
class MixedLinearHeadV2(nn.Module):
    def __init__(self, input_dim_list, max_embed_dim, linear_layer):
        super().__init__()
        self.input_dim_list = input_dim_list
        self.linear_layer = linear_layer
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, input_dim, linear_layer):
        weight = linear_layer.weight[:, :input_dim]
        bias = linear_layer.bias
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            weight, bias = self.sample_weights_and_bias(
                self.input_dim_list[weights_max], self.linear_layer)
            # pad weights and bias
            weight = weights[weights_max]*weight
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            # print(x.shape)
            for i in range(len(self.input_dim_list)):
                weight, bias = self.sample_weights_and_bias(
                self.input_dim_list[i], self.linear_layer)
                # pad weights and bias
                # print("Heeey",weight.shape)
                weight = F.pad(weights[i]*weight, (0, self.max_embed_dim -
                           weight.shape[-1]), "constant", 0)
                # print("Heeey",weight.shape)
                # print(bias.shape)
                weights_mix += weight
            out = F.linear(x, weights_mix, bias)

        return out
    
# test mixed linear head
'''if __name__ == "__main__":
    linear_layer = nn.Linear(10, 10)
    linear_layer.weight.data = torch.ones(10, 10)
    linear_layer.bias.data = torch.ones(10)
    input_dim_list = [6, 8, 10]
    max_embed_dim = 10
    mixed_linear_head = MixedLinearHeadV2(input_dim_list, max_embed_dim, linear_layer)
    x = torch.ones(1, 10)
    out = mixed_linear_head(x, [0.2, 0.7, 0.1])
    print(out)
    weight_true = 0
    weight_1 = torch.nn.functional.pad(0.2*linear_layer.weight[:, :6], (0, 4), "constant", 0)
    weight_2 = torch.nn.functional.pad(0.7*linear_layer.weight[:, :8], (0, 2), "constant", 0)
    weight_3 = torch.nn.functional.pad(0.1*linear_layer.weight[:, :10], (0, 0), "constant", 0)
    weight_true = weight_1 + weight_2 + weight_3
    out_true = F.linear(x, weight_true, linear_layer.bias)
    print(out_true)
    assert torch.allclose(out, out_true)
    out = mixed_linear_head(x, [0.2, 0.6, 0.2], use_argmax=True)
    weight_true = 0.6*linear_layer.weight[:, :8]
    out_true = F.linear(x[:,:8], weight_true, linear_layer.bias)
    print(out)
    print(out_true)
    assert torch.allclose(out, out_true)'''
