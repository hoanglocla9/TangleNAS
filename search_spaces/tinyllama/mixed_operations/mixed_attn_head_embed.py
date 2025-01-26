import torch.nn as nn
import torch.nn.functional as F
import torch
import math


from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_greater_or_equal_2_10


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MixedAttnHeadEmbed(torch.nn.Module):

    def __init__(self, num_heads_list, hidden_size_list, num_heads_kv, dropout,attn_dropout, rotary_emb, flash, bias):
        super().__init__()
        self.num_heads_list = num_heads_list
        self.max_head = max(num_heads_list)
        self.hidden_size_list = hidden_size_list
        self.max_hidden_size = max(hidden_size_list)
        self.max_head_dim = self.max_hidden_size // min(num_heads_list)
        self.num_heads_kv = num_heads_kv
        self.rotary_emb = rotary_emb
        self.dropout = dropout
        self.flash = flash
        self.attn_dropout = attn_dropout
        
        self.bias = bias

    def forward_attention(self, k , q, v, current_num_heads, position_ids, attention_mask):
        if self.flash:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # efficient attention using Flash Attention CUDA kernels
            y = _flash_attention_forward(
                q,
                k,
                v,
                attention_mask,
                self.T,
                position_ids=position_ids,
                dropout=self.dropout,
                sliding_window=None,
                use_top_left_mask= not is_flash_attn_greater_or_equal_2_10(),
                is_causal=True,
            )
            y = y.reshape(self.B, self.T, -1).contiguous()

        else:
            num_key_value_groups = current_num_heads // self.num_heads_kv
            k = repeat_kv(k, num_key_value_groups)
            v = repeat_kv(v, num_key_value_groups)

            attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : k.shape[-2]]
                attn_weights = attn_weights + causal_mask

            att = F.softmax(attn_weights, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(self.B, self.T, -1) # re-assemble all head outputs side by side
        
        return y

    def forward(self, x, weights, use_argmax=False):
        q_m, k_m, v_m, position_ids, position_embeddings, attention_mask  = x
        self.B = q_m.size(0)
        self.T = q_m.size(1)

        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            hidden_size_argmax_id = torch.div(weights_max, len(self.num_heads_list), rounding_mode='floor')
            num_heads_argmax_id = weights_max%len(self.hidden_size_list) if len(self.num_heads_list) > 1 else 0
            h = self.num_heads_list[num_heads_argmax_id]
            e = self.hidden_size_list[hidden_size_argmax_id]
            k = k_m[:,:,:e//h * self.num_heads_kv]
            q = q_m[:,:,:e]
            v = v_m[:,:,:e//h * self.num_heads_kv]
            k = k.view(self.B, self.T, self.num_heads_kv, e // h).transpose(1, 2)
            q = q.view(self.B, self.T, h, e // h).transpose(1, 2)
            v = v.view(self.B, self.T, self.num_heads_kv, e // h).transpose(1, 2)

            if position_embeddings is None:
                cos, sin = self.rotary_emb(v, e // h, position_ids)
            else:
                cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            k = weights[weights_max]*k
            q = weights[weights_max]*q
            v = weights[weights_max]*v
            out_mixture = self.forward_attention(k, q, v, self.num_heads_list[num_heads_argmax_id], position_ids, attention_mask)

        else:
            out_mixture = 0
            l = 0
            for i, h in enumerate(self.num_heads_list):
                k_mix = 0
                q_mix = 0
                v_mix = 0
                max_head_dim = self.max_hidden_size // h
                max_head_kv_dim = self.max_hidden_size // h * self.num_heads_kv

                for j, e in enumerate(self.hidden_size_list):
                    k = k_m[:,:,:e//h * self.num_heads_kv]
                    q = q_m[:,:,:e]
                    v = v_m[:,:,:e//h * self.num_heads_kv]
                    k = k.view(self.B, self.T, self.num_heads_kv, e // h).transpose(1, 2) # (B, nh, T, hs)
                    q = q.view(self.B, self.T, h, e // h).transpose(1, 2) # (B, nh, T, hs)
                    v = v.view(self.B, self.T, self.num_heads_kv, e // h).transpose(1, 2) # (B, nh, T, hs)
                    if position_embeddings is None:
                        cos, sin = self.rotary_emb(v,  e // h, position_ids)
                    else:
                        cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)

                    k_mix+= weights[l]*F.pad(k, (0,(max_head_kv_dim - k.shape[-1])), "constant", 0)
                    q_mix+= weights[l]*F.pad(q, (0,(max_head_dim - q.shape[-1])), "constant", 0)
                    v_mix+= weights[l]*F.pad(v, (0,(max_head_kv_dim - v.shape[-1])), "constant", 0)
                    l += 1

                out_curr = self.forward_attention(q, k, v, h, position_ids, attention_mask)
                out_curr = F.pad(out_curr, (0, self.max_hidden_size - out_curr.shape[-1]), "constant", 0) # pad to max embed dim
                out_mixture += out_curr
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
