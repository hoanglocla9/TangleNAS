import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class MixedAttnHeadEmbed(torch.nn.Module):

    def __init__(self, n_head_list, embed_dim_list, dropout, B, T, C, attn_dropout, flash, bias):
        super().__init__()
        self.n_head_list = n_head_list
        self.max_head = max(n_head_list)
        self.embed_dim_list = embed_dim_list
        self.max_embed_dim = max(embed_dim_list)
        self.max_head_dim = self.max_embed_dim // min(n_head_list)
        self.dropout = dropout
        self.flash = flash
        self.attn_dropout = attn_dropout
        self.B = B
        self.T = T
        self.C = C
        self.bias = bias
        self.mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).cuda()

    def forward_attention(self, k , q, v):
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:self.T,:self.T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #print(y.shape)
        y = y.transpose(1, 2).contiguous().view(self.B, self.T, -1) # re-assemble all head outputs side by side
        
        return y

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            input_dim_argmax_id = torch.div(weights_max, len(self.embed_dim_list), rounding_mode='floor')
            output_dim_argmax_id = weights_max%len(self.n_head_list)
            h = self.n_head_list[input_dim_argmax_id]
            e = self.embed_dim_list[output_dim_argmax_id]
            q_m, k_m, v_m  = x.split(e, dim=2)
            #print(h)
            #print(e)
            max_head_dim = self.max_embed_dim // h
            k = k_m[:,:,:e]
            q = q_m[:,:,:e]
            v = v_m[:,:,:e]
            k = weights[weights_max]*k.view(self.B, self.T, h, e // h).transpose(1, 2)
            q = weights[weights_max]*q.view(self.B, self.T, h, e // h).transpose(1, 2)
            v = weights[weights_max]*v.view(self.B, self.T, h, e // h).transpose(1, 2)
            #if max_head_dim > k.shape[-1]:
            #k = F.pad(k, (0, (max_head_dim - k.shape[-1])))
            #q = F.pad(q, (0, (max_head_dim - q.shape[-1])))
            #v = F.pad(v, (0, (max_head_dim - v.shape[-1])))
            #print(k.shape)
            #print(q.shape)
            #print(v.shape)
            out_mixture = self.forward_attention(k, q, v)
        else:
            q_m, k_m, v_m  = x.split(self.max_embed_dim, dim=2)
            out_mixture = 0
            l = 0
            for i, h in enumerate(self.n_head_list):
                k_mix = 0
                q_mix = 0
                v_mix = 0
                max_head_dim = self.max_embed_dim // h
                for j, e in enumerate(self.embed_dim_list):
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
                    k_mix+= weights[l]*F.pad(k, (0,(max_head_dim - k.shape[-1])), "constant", 0)
                    #print(k_mix.shape)
                    q_mix+= weights[l]*F.pad(q, (0,(max_head_dim - q.shape[-1])), "constant", 0)
                    v_mix+= weights[l]*F.pad(v, (0,(max_head_dim - v.shape[-1])), "constant", 0)
                    l += 1

                out_curr = self.forward_attention(k_mix, q_mix, v_mix)
                out_curr = F.pad(out_curr, (0, self.C - out_curr.shape[-1]), "constant", 0) # pad to max embed dim
                out_mixture += out_curr
                #print(out_mixture)
        return out_mixture
    

# test attn_head_embed
'''if __name__ == "__main__":
    head_list = [1,2,3]
    embed_dim_list = [6,12,18]
    dropout = 0.1
    B = 2
    T = 3
    C = 18
    attn_dropout = 0.1
    flash = False
    bias = True
    attn_head_embed = MixedAttnHeadEmbed(head_list, embed_dim_list, dropout, B, T, C, attn_dropout, flash, bias)
    x = torch.ones((B, T, 3*C))
    weights_head = [0.7, 0.2, 0.1]
    weights_embed = [0.1, 0.7, 0.2]
    # compute cross product of weights
    weights = []
    for i in range(len(weights_head)):
        for j in range(len(weights_embed)):
            weights.append(weights_head[i]*weights_embed[j])
    print(weights)
    print(sum(weights))
    out = attn_head_embed(x, weights)
    print(out.shape)
    print(out)'''

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
        self.mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).cuda()

    def forward_attention(self, k , q, v):
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:self.T,:self.T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #print(y.shape)
        y = y.transpose(1, 2).contiguous().view(self.B, self.T, -1) # re-assemble all head outputs side by side
        y = F.pad(y, (0, self.C - y.shape[-1]), "constant", 0) # pad to max embed dim
        return y

    def forward(self, x, weights, use_argmax=False):
        q_m, k_m, v_m  = x.split(self.max_embed_dim, dim=2)
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            output_dim_argmax_id = weights_max
            e = self.embed_dim_list[output_dim_argmax_id]
            h = self.num_heads
            max_head_dim = self.max_embed_dim // h
            k = k_m[:,:,:e]
            q = q_m[:,:,:e]
            v = v_m[:,:,:e]
            k = weights[weights_max]*k.view(self.B, self.T, h, e // h).transpose(1, 2)
            q = weights[weights_max]*q.view(self.B, self.T, h, e // h).transpose(1, 2)
            v = weights[weights_max]*v.view(self.B, self.T, h, e // h).transpose(1, 2)
            #if max_head_dim > k.shape[-1]:
            #print(k.shape)
            #print(q.shape)
            #print(v.shape)
            out_mixture = self.forward_attention(k, q, v)
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
            att = att.masked_fill(self.mask[:,:,:self.T,:self.T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #print(y.shape)
        y = y.transpose(1, 2).contiguous().view(self.B, self.T, -1) # re-assemble all head outputs side by side
        y = F.pad(y, (0, self.C - y.shape[-1]), "constant", 0) # pad to max embed dim
        return y

    def forward(self, x, weights, use_argmax=False):
        q_m, k_m, v_m  = x.split(self.max_embed_dim, dim=2)
        if use_argmax:
            weights_max = torch.argmax(weights, dim=-1)
            h = self.n_head_list[weights_max]
            head_dim = self.max_embed_dim // h
            k = k_m[:,:,:head_dim]
            q = q_m[:,:,:head_dim]
            v = v_m[:,:,:head_dim]
            k = weights[weights_max]*k.view(self.B, self.T, h, head_dim).transpose(1, 2)
            q = weights[weights_max]*q.view(self.B, self.T, h, head_dim).transpose(1, 2)
            v = weights[weights_max]*v.view(self.B, self.T, h, head_dim).transpose(1, 2)
            #if max_head_dim > k.shape[-1]:
            #print(k.shape)
            #print(q.shape)
            #print(v.shape)
            out_mixture = self.forward_attention(k, q, v)
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