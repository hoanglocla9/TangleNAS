import torch
import torch.nn as nn
import torch.nn.functional as F
import math


    

class MixedAttnEmbd(torch.nn.Module):

    def __init__(self, embed_dim_list, dropout, B, T, C, attn_dropout, flash, bias, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim_list = embed_dim_list
        self.max_embed_dim = max(embed_dim_list)
        self.max_head_dim = self.max_embed_dim // self.num_heads
        self.dropout = dropout
        self.flash = flash
        self.attn_dropout = attn_dropout
        self.B = B
        self.T = T
        self.C = C
        self.bias = bias

    def forward_attention(self, k , q, v):
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias.to(att.device)[:,:,:self.T,:self.T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(self.B, self.T, self.C) # re-assemble all head outputs side by side
        return y

    def forward(self, x, weights, use_argmax=False):
        q_m, k_m, v_m  = x.split(self.max_embed_dim, dim=2)
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            output_dim_argmax_id = weights_max
            e = self.embed_dim_list[output_dim_argmax_id]
            h = self.num_heads
            max_head_dim = self.max_embed_dim // h
            k = k_m[:,:,:e]
            q = q_m[:,:,:e]
            v = v_m[:,:,:e]
            k = k.view(self.B, self.T, h , e // h).transpose(1, 2)
            q = q.view(self.B, self.T, h, e // h).transpose(1, 2)
            v = v.view(self.B, self.T, h, e // h).transpose(1, 2)
            #if max_head_dim > k.shape[-1]:
            k = F.pad(k, (0, (max_head_dim - k.shape[-1])))
            q = F.pad(q, (0, (max_head_dim - q.shape[-1])))
            v = F.pad(v, (0, (max_head_dim - v.shape[-1])))
            #print(k.shape)
            #print(q.shape)
            #print(v.shape)
            out_mixture = weights[weights_max]*self.forward_attention(k, q, v)
        else:
            h = self.num_heads
            max_head_dim = self.max_embed_dim // h
            k_mix = 0
            q_mix = 0
            v_mix = 0
            for j, e in enumerate(self.embed_dim_list):
                k = k_m[:,:,:e]
                q = q_m[:,:,:e]
                v = v_m[:,:,:e]
                k = k.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                q = q.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                v = v.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                k_mix+= weights[j]*F.pad(k, (0,(max_head_dim - k.shape[-1])), "constant", 0)
                q_mix+= weights[j]*F.pad(q, (0,(max_head_dim - q.shape[-1])), "constant", 0)
                v_mix+= weights[j]*F.pad(v, (0,(max_head_dim - v.shape[-1])), "constant", 0)
            out_mixture = self.forward_attention(k_mix, q_mix, v_mix)
        return out_mixture

class MixedAttnHead(torch.nn.Module):

    def __init__(self, n_head_list, dropout, B, T, C, attn_dropout, flash, bias, max_embed_dim):
        super().__init__()
        self.n_head_list = n_head_list
        self.max_head = max(n_head_list)
        self.max_embed_dim = max_embed_dim
        self.max_head_dim = self.max_embed_dim // min(n_head_list)
        self.dropout = dropout
        self.flash = flash
        self.attn_dropout = attn_dropout
        self.B = B
        self.T = T
        self.C = C
        self.bias = bias

    def forward_attention(self, k , q, v):
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias.to(att.device)[:,:,:self.T,:self.T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(self.B, self.T, self.C) # re-assemble all head outputs side by side
        return y

    def forward(self, x, weights, use_argmax=False):
        q_m, k_m, v_m  = x.split(self.max_embed_dim, dim=2)
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            h = self.n_head_list[weights_max]
            head_dim = self.max_embed_dim // h
            k = k_m[:,:,:head_dim]
            q = q_m[:,:,:head_dim]
            v = v_m[:,:,:head_dim]
            k = k.view(self.B, self.T, h, head_dim).transpose(1, 2)
            q = q.view(self.B, self.T, h, head_dim).transpose(1, 2)
            v = v.view(self.B, self.T, h, head_dim).transpose(1, 2)
            #if max_head_dim > k.shape[-1]:
            k = F.pad(k, (0, (self.max_head_dim - k.shape[-1])))
            q = F.pad(q, (0, (self.max_head_dim - q.shape[-1])))
            v = F.pad(v, (0, (self.max_head_dim - v.shape[-1])))
            #print(k.shape)
            #print(q.shape)
            #print(v.shape)
            out_mixture = weights[weights_max]*self.forward_attention(k, q, v)
        else:
            out_mixture = 0
            l = 0
            for i, h in enumerate(self.n_head_list):
                e = self.max_embed_dim // h
                k = k_m[:,:,:e]
                q = q_m[:,:,:e]
                v = v_m[:,:,:e]
                k = k.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                q = q.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                v = v.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                #print(k.shape)
                #if max_head_dim > k.shape[-1]:
                #print(weights[l])
                #print(F.pad(k, (0,(max_head_dim - k.shape[-1])), "constant", 0).shape)
                k = weights[l]*F.pad(k, (0,(self.max_head_dim - k.shape[-1])), "constant", 0)
                #print(k_mix.shape)
                q = weights[l]*F.pad(q, (0,(self.max_head_dim - q.shape[-1])), "constant", 0)
                v = weights[l]*F.pad(v, (0,(self.max_head_dim - v.shape[-1])), "constant", 0)
                l += 1
                out_mixture += self.forward_attention(k, q, v)
        return out_mixture

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
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            weight, bias = self.sample_weights_and_bias(
                    self.input_dim_list[weights_max], self.linear_layer)
            if not self.reverse:
                weight = weights[weights_max]*F.pad(
                    weight, (0, self.max_in_dim-weight.shape[-1]), "constant", 0)
                
            else:
                weight = weights[weights_max]*F.pad(
                    weight, (0,0,0, self.max_in_dim-weight.shape[-2]), "constant", 0)
                if bias is not None:
                    bias = weights[weights_max]*F.pad(bias, (0, self.max_in_dim -
                            bias.shape[-1]), "constant", 0)
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
                    bias_mix += weights[k]*bias
                else:
                    bias_mix = None
                k = k+1
            out = F.linear(x, weights_mix, bias_mix)

        return out