from transformers.models.mamba2.modeling_mamba2 import Mamba2PreTrainedModel, Mamba2Output
from transformers.activations import  ACT2FN

from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput

from torch.nn import CrossEntropyLoss

from typing import Optional, Union, Tuple
from torch import nn
import torch, math
from mamba_ssm.modules.mha import _update_kv_cache

from einops import rearrange

import inspect
import torch.nn.functional as F
try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


from optimizers.mixop.entangle import EntangledOp
from dataclasses import dataclass
from mixed_operations.mixed_conv1d import MixedConv1dV2, MixedConv1dV2_MHA
from mixed_operations.mixed_rms_norm import MixedRMSNormGatedV2
from mixed_operations.mixed_linear_mamba2 import MixedLinearV2_InProj, MixedLinearV2_OutProj, MixedLinearV2_InProj_MHA, MixedLinearV2_OutProj_MHA
from mixed_operations.mixed_embedding import MixedEmbeddingV2
import itertools
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

from mixed_operations.mixed_linear_head import MixedLinearHeadV2
from optimizers.optim_factory import get_mixop, get_sampler

def get_entangle_ops(op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

def get_entangle_ops_combi(op, choices1, choices2, op_name):
    choices = list(itertools.product(choices1, choices2))
    return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states

class EntangleHybridMamba2Config(PretrainedConfig):
    model_type = "mamba2"
    def __init__(
        self,
        num_heads_list:list = [16, 24, 32],
        vocab_size:int = 50304,
        hidden_size_list:list = [384, 768, 1024],
        state_size:int = 128,
        num_hidden_layers_list:list = [8, 9, 10],
        layer_norm_epsilon=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        expand=2,
        conv_kernel=4,
        n_groups=1,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual=False,
        use_cache=True,
        rms_norm=True,
        chunk_size=256,
        tie_word_embeddings=True,
        attn_layer_idxs=[],
        attn_num_heads_list:list = [16, 24, 32],
        attn_embed_dim_list:list = [384, 768, 1024],
        max_seqlen=2048,
        mixop=None,
        device='cuda',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size_list = hidden_size_list
        self.state_size = state_size
        self.num_hidden_layers_list = num_hidden_layers_list
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.device=device
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.mixop=mixop
        # self.time_step_rank = math.ceil(max(self.hidden_size_list) / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.n_groups = n_groups
        self.num_heads_list = num_heads_list
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.tie_word_embeddings = tie_word_embeddings
        self.attn_layer_idxs = attn_layer_idxs
        self.max_seqlen = max_seqlen
        
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class HybridMamba2Cache:
    """
    Arguments:
        config: Mamba2Config
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self, config: EntangleHybridMamba2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.max_seqlen = config.max_seqlen
        self.max_batch_size = batch_size
        self.batch_size_offset = 0
        self.lengths_per_sample = None
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * max(config.hidden_size_list))
        self.key_value_memory_dict = {}

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.n_groups * config.state_size,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(max(config.num_hidden_layers_list)) if i not in config.attn_layer_idxs
        }
        for i in config.attn_layer_idxs:
            self.conv_states[i] = torch.zeros(
                batch_size,
                max(config.hidden_size_list)//min(config.num_heads_list) * 3 * max(config.num_heads_list),
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
        self.ssm_states = {
            i: torch.zeros(
                batch_size, max(config.num_heads_list), int(max(config.hidden_size_list) * config.expand)//min(config.num_heads_list), config.state_size, device=device, dtype=dtype
            )
            for i in range(max(config.num_hidden_layers_list)) if i not in config.attn_layer_idxs  #
        }
        ### TODO: fix for hybrid models with attention layers.
        self.kv_states = {
            i: torch.zeros(
                batch_size, config.max_seqlen, 2, max(config.num_heads_list), int(max(config.hidden_size_list))//min(config.num_heads_list), device=device, dtype=dtype
            )
            for i in config.attn_layer_idxs
        } #  * config.expand

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init:bool
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states[layer_idx].device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states[layer_idx].device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]
    
    def update_kv_cache(self,layer_idx: int, kv: torch.Tensor):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        # Pre-allocate memory for key-values for inference.
        num_heads, head_dim = kv.shape[-2:]
        assert layer_idx in self.kv_states
        assert num_heads == self.kv_states[layer_idx].shape[3] and head_dim == self.kv_states[layer_idx].shape[4]
        kv_cache = self.kv_states[layer_idx]
        # Adjust key and value for inference
        batch_start = self.batch_size_offset
        batch_end = batch_start + kv.shape[0]
        sequence_start = self.seqlen_offset
        sequence_end = sequence_start + kv.shape[1]
        assert batch_end <= kv_cache.shape[0]
        assert sequence_end <= kv_cache.shape[1]
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        self.kv_states[layer_idx].zero_()
        self.kv_states[layer_idx] += kv_cache
        return self.kv_states[layer_idx] 

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

        self.conv_states.zero_()
        self.ssm_states.zero_()
        self.kv_states.zero_()

class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        # embed_dim,
        # num_heads,
        config: EntangleHybridMamba2Config=None, 
        mixop: EntangledOp=None,    
        num_heads_kv=None,
        head_dim=None,  # If None, use embed_dim // num_heads
        mlp_dim=0,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,
        layer_idx:int=None,
        d_conv=0,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        # device='cuda',
        # dtype=torch.float16,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        # factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # self.embed_dim = embed_dim
        self.mixop = mixop
        self.embed_dim_list = config.hidden_size_list
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.causal = causal

        # self.num_heads = num_heads
        self.num_heads_list = config.num_heads_list
        self.num_heads_kv_list = self.num_heads_list
        # self.num_heads_kv =  num_heads_kv if num_heads_kv is not None else num_heads
        
        # num_heads == num_heads_kv for now
        # if head_dim is None:
        #     assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.max_head_dim =  max(self.embed_dim_list)//min(self.num_heads_list)   # head_dim if head_dim is not None else self.embed_dim // num_heads

        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        max_qkv_dim = self.max_head_dim * 3 * (max(self.num_heads_list)) #  + 2 * self.num_heads_kv
        max_out_dim = self.max_head_dim * max(self.num_heads_list)
        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary requires flash_attn to be installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        in_proj = nn.Linear(max(self.embed_dim_list), max_qkv_dim + self.mlp_dim, bias=qkv_proj_bias ) # factory_kwargs
        self.in_proj_mix_op = MixedLinearV2_InProj_MHA(self.embed_dim_list, self.num_heads_list, mlp_dim=mlp_dim, linear_layer=in_proj)
        self.in_proj_mix_op_list = get_entangle_ops_combi(self.in_proj_mix_op, self.embed_dim_list, self.num_heads_list, "in_proj_mha")

        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                max_qkv_dim, max_qkv_dim, kernel_size=self.d_conv, padding=self.d_conv - 1, groups=max_qkv_dim,
                # **factory_kwargs
            )
            self.conv1d_mix_op = MixedConv1dV2_MHA(self.embed_dim_list, self.num_heads_list, self.conv1d)
            self.conv1d_mix_op_list = get_entangle_ops_combi(self.conv1d_mix_op, self.embed_dim_list, self.num_heads_list, "conv1d_mha")
        
        
        out_proj = nn.Linear(max_out_dim + self.mlp_dim // 2, max(self.embed_dim_list), bias=out_proj_bias)  # , **factory_kwargs
        self.out_proj_mix_op = MixedLinearV2_OutProj_MHA(self.embed_dim_list, self.num_heads_list, mlp_dim=mlp_dim, linear_layer=out_proj)
        self.out_proj_mix_op_list = get_entangle_ops_combi(self.out_proj_mix_op, self.embed_dim_list, self.num_heads_list, "out_proj_mha")

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.conv1d.weight.dtype if dtype is None else dtype
        device = self.conv1d.weight.device
        if self.d_conv > 0:
            conv_state = torch.zeros(
                batch_size, self.max_head_dim * 3 * (max(self.num_heads_list)), self.d_conv, device=device, dtype=dtype # self.conv1d.weight.shape[0]
            )
        else:
            conv_state = None
        kv_cache = torch.empty(
            batch_size, max_seqlen, 2, max(self.num_heads_list), self.max_head_dim, dtype=dtype, device=device,
        )
        return kv_cache, conv_state

    def _update_kv_cache(self, kv, cache_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return cache_params.update_kv_cache(self.layer_idx, kv)

    def _apply_rotary_update_kvcache_attention(self, q, kv, cache_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert cache_params is not None and cache_params.seqlen_offset > 0
        if self.rotary_emb_dim > 0:
            self.rotary_emb._update_cos_sin_cache(
                cache_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache, _ = cache_params.key_value_memory_dict[self.layer_idx]
        kv_cache = kv_cache[:batch]
        cache_seqlens = (
            cache_params.lengths_per_sample[:batch]
            if cache_params.lengths_per_sample is not None
            else cache_params.seqlen_offset
        )
        assert flash_attn_with_kvcache is not None, "flash_attn must be installed"
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            rotary_interleaved= self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
        )
        return context

    def _update_kvcache_attention(self, q, kv, cache_params, attention_mask=None):
        """Write kv to cache_params, then do attention"""
        ## TODO: Need to support attetion mask

        # if attention_mask is not None:
        #     batch_size = q.shape[0]
        #     query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
        #         query_states, key_states, value_states, attention_mask, query_length
        #     )

        if (
            cache_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, cache_params)
            k, v = kv.unbind(dim=-3)
            k = torch.repeat_interleave(k, dim=2, repeats=1) # self.num_heads == self.num_heads_kv)
            v = torch.repeat_interleave(v, dim=2, repeats=1) # self.num_heads // self.num_heads_kv
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
            ).transpose(1, 2)
        else:
            batch = q.shape[0]
            kv_cache, _ = cache_params.kv_states[self.layer_idx]
            kv_cache = kv_cache[:batch]
            cache_seqlens = (
                # cache_params.lengths_per_sample[:batch]
                # if cache_params.lengths_per_sample is not None
                # else 
                cache_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

    def get_mixed_conv1d_weights(self, embeded_dim_weights, num_heads_weights):
        weights_mix = 0
        bias_mix = 0
        max_n_channels = max(self.embed_dim_list)//min(self.num_heads_list) * 3 * (max(self.num_heads_list))
        for i in range(len(self.embed_dim_list)):
            for j in range(len(self.num_heads_list)):
                n_channels = self.embed_dim_list[i] // self.num_heads_list[j] * 3 * self.num_heads_list[j]
                weight = self.conv1d.weight[:n_channels,:,:]
                if self.conv1d.bias is None:
                    bias = None
                else:
                    bias = self.conv1d.bias[:n_channels]
                    # pad weights and bias
                weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, max_n_channels - weight.size(0)), "constant", 0)
                if bias is None:
                    bias = None
                else:
                    bias = torch.nn.functional.pad(bias, (0, max_n_channels - bias.shape[0]), "constant", 0)
                weights_mix += embeded_dim_weights[i] * num_heads_weights[j] *   weight
                if bias is not None:
                    bias_mix += embeded_dim_weights[i] * num_heads_weights[j] * bias
                else:
                    bias_mix = None

        return weights_mix, bias_mix

    def forward(self, x, 
                cache_params:HybridMamba2Cache=None, cache_position: Optional[torch.LongTensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                arch_params: dict = None, 
                use_argmax: Optional[bool] = False):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cache_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        residual = x
        if cache_params is not None and self.layer_idx not in cache_params.key_value_memory_dict:
            cache_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                x.shape[0], cache_params.max_seqlen, dtype=x.dtype
            )
        seqlen_offset = (
            0
            if cache_params is None
            else (
                # cache_params.lengths_per_sample
                # if cache_params.lengths_per_sample is not None
                # else 
                cache_params.seqlen_offset
            )
        )
        rotary_max_seqlen = cache_params.max_seqlen if cache_params is not None else None
        # qkv = self.in_proj(x)
        qkv = self.mixop.forward(x,  [arch_params["hidden_size"], arch_params["num_heads"][self.layer_idx]], self.in_proj_mix_op_list, combi=True)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)

        if self.d_conv > 0:
            # The inference code for conv1d is pretty messy, should clean it up
            conv1d_mix, bias_mix = self.get_mixed_conv1d_weights(arch_params["hidden_size"], arch_params["num_heads"][self.layer_idx])
            if (cache_params is None or cache_params.seqlen_offset == 0):
                # if causal_conv1d_fn is None:
                qkv = rearrange(
                    self.mixop.forward(rearrange(qkv, "b s d -> b d s"), [arch_params["hidden_size"], arch_params["num_heads"][self.layer_idx]], 
                        self.conv1d_mix_op_list, combi=True)[..., :-(self.d_conv - 1)]   , "b d s -> b s d"
                ).contiguous() # conv1d(rearrange(qkv, "b s d -> b d s"))
                # else:
                # qkv = causal_conv1d_fn(
                #     qkv.transpose(1, 2),
                #     rearrange(conv1d_mix, "d 1 w -> d w"),
                #     bias_mix
                # ).transpose(1, 2)
                if cache_params is not None:
                    conv_state = cache_params.conv_states[self.layer_idx]
                    # If we just take qkv[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    qkv_t = rearrange(qkv, "b l d -> b d l")
                    conv_state.copy_(F.pad(qkv_t, (self.d_conv - qkv_t.shape[-1], 0)))  # Update state (B D W)
            else:
                conv_state = cache_params.conv_states[self.layer_idx]
                assert qkv.shape[1] == 1, "Only support decoding with 1 token at a time for now"
                qkv = qkv.squeeze(1)
                # Conv step
                # if causal_conv1d_update is None:
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
                conv_state[:, :, -1] = qkv
                qkv = torch.sum(conv_state * rearrange(conv1d_mix, "d 1 w -> d w"), dim=-1)  # (B D)

                if bias_mix is not None:
                    qkv = qkv + bias_mix
                # else:
                #     qkv = causal_conv1d_update(
                #         qkv,
                #         conv_state,
                #         rearrange(self.conv1d.weight, "d 1 w -> d w"),
                #         self.conv1d.bias
                #     )
                qkv = qkv.unsqueeze(1)

        q, kv = qkv.split([max(self.num_heads_list) * self.max_head_dim, max(self.num_heads_kv_list) * 2 * self.max_head_dim], dim=-1)
        q = rearrange(q, "... (h d) -> ... h d", d=self.max_head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.max_head_dim)
        if (
            cache_params is None
            or cache_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if cache_params is None:
                k, v = kv.unbind(dim=-3)
                k = torch.repeat_interleave(k, dim=2, repeats=1)
                v = torch.repeat_interleave(v, dim=2, repeats=1)
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                ).transpose(1, 2)
            else:
                context = self._update_kvcache_attention(q, kv, cache_params, attention_mask=attention_mask)
        else:
            context = self._apply_rotary_update_kvcache_attention(q, kv, cache_params, attention_mask=attention_mask)
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.mixop.forward(context,  [arch_params["hidden_size"], arch_params["num_heads"][self.layer_idx]], self.out_proj_mix_op_list, combi=True)
        out = out + residual
        return out
    

class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)

class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: EntangleHybridMamba2Config, mixop, layer_idx: int):
        super().__init__()
        self.num_heads_list = config.num_heads_list
        self.hidden_size_list = config.hidden_size_list
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        # self.intermediate_size = int(config.expand * max(self.hidden_size_list))
        # self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups

        ## TODO: Double check head_dim = hidden_size * expand // num_heads
        self.max_head_dim = config.expand * max(self.hidden_size_list)//max(self.num_heads_list)
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max
        self.mixop = mixop
        self.expand = config.expand

        # self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        ### TODO: Double check
        self.conv1d = nn.Conv1d(
            in_channels=int(config.expand * max(self.hidden_size_list)) + 2 * self.n_groups * self.ssm_state_size,
            out_channels=int(config.expand * max(self.hidden_size_list)) + 2 * self.n_groups * self.ssm_state_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=int(config.expand * max(self.hidden_size_list)) + 2 * self.n_groups * self.ssm_state_size,
            padding=config.conv_kernel - 1,
        )
        conv1d_dim_list = [int(config.expand * hidden_size) + 2 * self.n_groups * self.ssm_state_size for hidden_size in self.hidden_size_list]
        self.conv1d_mix_op = MixedConv1dV2(conv1d_dim_list, self.conv1d)
        self.conv1d_mix_op_list = get_entangle_ops(self.conv1d_mix_op, conv1d_dim_list, "conv1d_mamba2")
        
        # projection of the input hidden states
        in_proj = nn.Linear(
            max(self.hidden_size_list),
            2*(int(config.expand * max(self.hidden_size_list)) + self.n_groups * self.ssm_state_size) + max(self.num_heads_list),
            bias=config.use_bias,
        )

        self.in_proj_mix_op = MixedLinearV2_InProj(self.hidden_size_list, self.num_heads_list, expand=self.expand, n_groups=self.n_groups, ssm_state_size=self.ssm_state_size, linear_layer=in_proj)
        self.in_proj_mix_op_list = get_entangle_ops_combi(self.in_proj_mix_op, self.hidden_size_list, self.num_heads_list, "in_proj_mamba2")
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(max(self.num_heads_list)))
        # self.dt_bias_mix_op_list = self.get_entangle_ops(self.in_proj_mix_op, self.hidden_size_list, "in_proj_mamba2")
        
        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, max(self.num_heads_list) + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        norm_f = MambaRMSNormGated(int(config.expand * max(self.hidden_size_list)), eps=self.layer_norm_epsilon)
        self.norm_f_op = MixedRMSNormGatedV2([int(config.expand * hidden_size) for hidden_size in self.hidden_size_list], int(config.expand * max(self.hidden_size_list)), norm_f)
        self.norm_f_list = get_entangle_ops(self.norm_f_op, [int(config.expand * hidden_size) for hidden_size in self.hidden_size_list], f"mix_norm_f")

        self.D = nn.Parameter(torch.ones(max(self.num_heads_list)))
        self.D._no_weight_decay = True

        out_proj = nn.Linear(int(config.expand * max(self.hidden_size_list)), max(self.hidden_size_list), bias=config.use_bias)
        self.out_proj_mix_op = MixedLinearV2_OutProj(self.hidden_size_list, expand=config.expand, linear_layer=out_proj)
        self.out_proj_mix_op_list = get_entangle_ops(self.out_proj_mix_op, self.hidden_size_list, "out_proj_mamba2")

        self.use_bias = config.use_bias

    def get_mixed_conv1d_weights(self, param_weights):
        weights_mix = 0
        bias_mix = 0
        n_channels_list = [int(self.expand * hidden_size) + 2 * self.n_groups * self.ssm_state_size for hidden_size in self.hidden_size_list]
        max_n_channels = max(n_channels_list)
        for i, n_channels in enumerate(n_channels_list):
            weight = self.conv1d.weight[:n_channels,:,:]
            if self.conv1d.bias is None:
                bias = None
            else:
                bias = self.conv1d.bias[:n_channels]
            # pad weights and bias
            weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, max_n_channels - weight.size(0)), "constant", 0)
            if bias is None:
                bias = None
            else:
                bias = torch.nn.functional.pad(bias, (0, max_n_channels - bias.shape[0]), "constant", 0)
            weights_mix += param_weights[i]*weight
            if bias is not None:
                bias_mix += param_weights[i]*bias
            else:
                bias_mix = None
        return weights_mix, bias_mix

        
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        arch_params: dict = None, 
        use_argmax: Optional[bool] = False
    ):
        if use_argmax:
            # hidden_size_argmax_id = torch.argmax(torch.tensor(arch_params["hidden_size"]), dim=-1)
            # num_heads_argmax_id = torch.argmax(torch.tensor(arch_params['num_heads'][self.layer_idx]), dim=-1)
            raise NotImplementedError("Argmax not implemented for Mamba2")
            
        else:
            # 1. Gated MLP's linear projection
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
            projected_states = self.mixop.forward(hidden_states, [arch_params["hidden_size"], arch_params["num_heads"][self.layer_idx]], self.in_proj_mix_op_list, combi=True)


            # Set up dimensions for reshapes later
            batch_size, seq_len, _ = hidden_states.shape
            groups_time_state_size = self.n_groups * self.ssm_state_size
            intermediate_size = int(self.expand * max(self.hidden_size_list))
            conv_dim = intermediate_size + 2 * self.n_groups * self.ssm_state_size
            d_mlp = (
                projected_states.shape[-1]
                - 2 * intermediate_size
                - 2 * self.n_groups * self.ssm_state_size
                - max(self.num_heads_list)
            ) // 2
            _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [d_mlp, d_mlp, intermediate_size, conv_dim, max(self.num_heads_list)], dim=-1
            )
            mixed_conv1d_weights, mixed_conv1d_bias = self.get_mixed_conv1d_weights(arch_params['hidden_size'])
            # Single step calculations via cache
            if cache_params is not None and cache_position is not None and cache_position[0] > 0:
                # 2. Convolution sequence transformation
                hidden_states_B_C = causal_conv1d_update(
                    hidden_states_B_C,
                    cache_params.conv_states[self.layer_idx],
                    mixed_conv1d_weights.squeeze(1),
                    mixed_conv1d_bias,
                    self.activation,
                )

                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                # 3. SSM transformation
                B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups) # (batch_size, n_groups, ssm_state_size)
                C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups) # (batch_size, n_groups, ssm_state_size)
                hidden_states_reshaped = hidden_states.view(batch_size, max(self.num_heads_list), self.max_head_dim)
                dt_bias = self.dt_bias[:, None, ...].expand(-1, self.max_head_dim)
                D = self.D[:, None, ...].expand(-1, self.max_head_dim)
                A = -torch.exp(self.A_log.float())  # (nheads,)
                A = A[:, None, ...][:, :, None].expand(-1, self.max_head_dim, self.ssm_state_size).to(dtype=torch.float32) # (nheads, head_dim, ssm_state_size)
                dt = dt.squeeze(2)  # (batch_size, seq_len, nheads)        
                A_mix, dt_mix, D_mix, dt_bias_mix = 0,0,0,0
                for i, num_heads in enumerate(self.num_heads_list):
                    # Compute dt_mix
                    sampled_dt = dt[:,:,:num_heads]
                    sampled_dt = torch.nn.functional.pad(sampled_dt, (0, max(self.num_heads_list)-num_heads, 0, 0), "constant", 0)
                    dt_mix += arch_params['num_heads'][self.layer_idx][i]*sampled_dt

                    for j, hidden_size in enumerate(self.hidden_size_list):
                        head_dim = self.expand * hidden_size // num_heads
                        ## Compute A_mix
                        sampled_A = A[:num_heads,:head_dim,:]
                        sampled_A = torch.nn.functional.pad(sampled_A, (0, 0, 0, self.max_head_dim-head_dim, 0, max(self.num_heads_list)-num_heads), "constant", 0)
                        A_mix += arch_params['num_heads'][self.layer_idx][i] * arch_params['hidden_size'][j] * sampled_A
                            
                        ## Compute dt_bias_mix
                        # (nheads, head_dim)
                        sampled_dt_bias = dt_bias[:num_heads,:head_dim]
                        sampled_dt_bias = torch.nn.functional.pad(sampled_dt_bias, (0, self.max_head_dim-head_dim, 0, max(self.num_heads_list)-num_heads), "constant", 0)
                    
                        dt_bias_mix += arch_params['num_heads'][self.layer_idx][i] * arch_params['hidden_size'][j] * sampled_dt_bias

                        ## Compute D_mix
                        sampled_D = D[:num_heads,:head_dim]
                        sampled_D = torch.nn.functional.pad(sampled_D, (0, self.max_head_dim-head_dim, 0, max(self.num_heads_list)-num_heads), "constant", 0)
                        D_mix += arch_params['num_heads'][self.layer_idx][i] * arch_params['hidden_size'][j] * sampled_D


                hidden_states = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states_reshaped,
                    dt_mix,
                    A_mix,
                    B,
                    C,
                    D_mix,
                    z=None,
                    dt_bias=dt_bias_mix,
                    dt_softplus=True,
                )
                hidden_states = hidden_states.view(batch_size, max(self.num_heads_list) * self.max_head_dim)
                hidden_states = self.mixop.forward(hidden_states, arch_params["hidden_size"], self.norm_f_list, gate=gate)

                # 4. Final linear projection
                out = self.mixop.forward(hidden_states, arch_params["hidden_size"], self.out_proj_mix_op_list)[:, None, ...]
            else:
                A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
                dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}
                
                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = nn.functional.pad(
                        hidden_states_B_C_transposed,
                        (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True
                    )

                if self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.mixop.forward(hidden_states_B_C.transpose(1, 2), arch_params['hidden_size'], self.conv1d_mix_op_list)[..., :seq_len].transpose(1, 2)
                    ) # self.conv1d(hidden_states_B_C.transpose(1, 2))
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=mixed_conv1d_weights.squeeze(1),
                        bias=mixed_conv1d_bias,
                        activation=self.activation,
                    ).transpose(1, 2)
                    
                hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                A_mix, dt_mix, D_mix, dt_bias_mix = 0,0,0,0
                for i, num_heads in enumerate(self.num_heads_list):
                    # Compute dt_mix
                    sampled_dt = dt[:,:,:num_heads]
                    sampled_dt = torch.nn.functional.pad(sampled_dt, (0, max(self.num_heads_list)-num_heads, 0, 0), "constant", 0)
                    dt_mix += arch_params['num_heads'][self.layer_idx][i]*sampled_dt
                    ## Compute A_mix
                    sampled_A = A[:num_heads]
                    sampled_A = torch.nn.functional.pad(sampled_A, (0, max(self.num_heads_list)-num_heads), "constant", 0)
                    A_mix += arch_params['num_heads'][self.layer_idx][i] * sampled_A
                            
                    ## Compute dt_bias_mix
                    sampled_dt_bias = self.dt_bias[:num_heads]
                    sampled_dt_bias = torch.nn.functional.pad(sampled_dt_bias, (0, max(self.num_heads_list)-num_heads), "constant", 0)
                    
                    dt_bias_mix += arch_params['num_heads'][self.layer_idx][i] * sampled_dt_bias

                    ## Compute D_mix
                    sampled_D = self.D[:num_heads]
                    sampled_D = torch.nn.functional.pad(sampled_D, (0, max(self.num_heads_list)-num_heads), "constant", 0)
                    D_mix += arch_params['num_heads'][self.layer_idx][i]  * sampled_D

                # 3. SSM transformation
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, max(self.num_heads_list), self.max_head_dim),
                    dt_mix,
                    A_mix,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=D_mix,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=dt_bias_mix,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                # Init cache
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output  = self.mixop.forward(scan_output, arch_params["hidden_size"], self.norm_f_list, gate=gate)

                # 4. Final linear projection
                out = self.mixop.forward(scan_output, arch_params["hidden_size"], self.out_proj_mix_op_list)
            
        
        return out

    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        arch_params: dict=None,
        use_argmax: Optional[bool] = False
    ):
        return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask, arch_params=arch_params,use_argmax=use_argmax)

class Mamba2Block(nn.Module):
    def __init__(self, config, mixop, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.hidden_size_list = config.hidden_size_list
        self.norm_f = MambaRMSNormGated(max(self.hidden_size_list), eps=config.layer_norm_epsilon)
        self.norm_f_op = MixedRMSNormGatedV2(self.hidden_size_list, max(self.hidden_size_list), self.norm_f)
        self.norm_f_list = get_entangle_ops(self.norm_f_op, self.hidden_size_list, f"block_norm_f") # _{layer_idx}

        self.mixer = Mamba2Mixer(config, mixop, layer_idx=layer_idx)
        self.mixop = mixop

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        arch_params: dict = None,
    ):
        residual = hidden_states
        hidden_states = self.mixop.forward(hidden_states.to(dtype=self.norm_f.weight.dtype), 
                        arch_params["hidden_size"], self.norm_f_list)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask, arch_params=arch_params
        )
        hidden_states = residual + hidden_states
        return hidden_states

    
class Mamba2AttentionModel(Mamba2PreTrainedModel):
    def __init__(self, config, mixop,  max_n_layers, hidden_size_list, max_hidden_size):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, max_hidden_size).to(config.device)
        self.embedding_table_op =  MixedEmbeddingV2(hidden_size_list, max_embed_dim=max_hidden_size, embedding=self.embeddings)
        self.embedding_table_list = get_entangle_ops(self.embedding_table_op, hidden_size_list, "embedding_table")
        tmp_layers = []
        for idx in range(max_n_layers):
            ####### TODO: Fix for hybrid models, Currently the model only consists of Mamba blocks
            if idx in config.attn_layer_idxs:
                tmp_layers.append(MHA(config, d_conv=config.conv_kernel,  causal=True, mixop=mixop, layer_idx=idx)) # 
            else:
                tmp_layers.append(Mamba2Block(config, mixop, layer_idx=idx))
        self.layers = nn.ModuleList(tmp_layers) # [Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]

        norm_f = MambaRMSNormGated(max_hidden_size, eps=config.layer_norm_epsilon)

        self.norm_f_op = MixedRMSNormGatedV2(hidden_size_list, max_hidden_size, norm_f)
        self.norm_f_list = get_entangle_ops(self.norm_f_op, hidden_size_list, "last_norm_f")

        self.mixop = mixop
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMamba2Cache] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        arch_params:dict =None,
    ) -> Union[Tuple, Mamba2Output]:
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            # inputs_embeds = self.embeddings(input_ids)
            inputs_embeds = self.mixop.forward(input_ids, arch_params["hidden_size"], self.embedding_table_list)


        if use_cache:
            if cache_params is None:
                cache_params = HybridMamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)

            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = None
        
        depth_output_list = []
        i = 0
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
                arch_params=arch_params
            )
            if i+1 in self.config.num_hidden_layers_list:
                depth_output_list.append(hidden_states)
            i += 1

        hidden_states = self.mixop.forward_depth(
            depth_output_list, arch_params["num_hidden_layers"])
        
        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.mixop.forward(hidden_states, arch_params['hidden_size'], self.norm_f_list) 


        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )
    

@dataclass
class HybridMamba2CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[HybridMamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class HybridMamba2ForCausalLM(Mamba2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.layer_list = config.num_hidden_layers_list
        self.max_n_layers = max(self.layer_list)

        self.num_head_list = config.num_heads_list
        self.max_head = max(self.num_head_list)

        self.hidden_size_list = config.hidden_size_list
        self.max_hidden_size = max(self.hidden_size_list)
        
        self.mixop = get_mixop(config.mixop, use_we_v2=True) # use_we_v2=True
        self.sampler = get_sampler(config.mixop)

        self.backbone = Mamba2AttentionModel(config, self.mixop, self.max_n_layers, self.hidden_size_list, self.max_hidden_size)

        
        self.lm_head = nn.Linear(max(self.hidden_size_list), config.vocab_size, bias=False)
        self.lm_head_op = MixedLinearHeadV2(self.hidden_size_list, self.max_hidden_size, self.lm_head)
        self.lm_head_list = get_entangle_ops(self.lm_head_op, self.hidden_size_list, "lm_head")
        # Initialize weights and apply final processing
        self.post_init()
        self._init_arch_parameters()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[HybridMamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`

        if inputs_embeds is not None:
            past_len = inputs_embeds.shape[1] + input_ids.shape[1]
        else:
            past_len = input_ids.shape[1]
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            # how do we detect that we are in decoding without cache?
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1][..., None]
                attention_mask = attention_mask[:, -1][..., None]
            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, past_len, device=input_ids.device)
                # if the cache is not used, we also do have to extend the attention mask here
                # TODO there is likely a cleverer way to do this
                extended_mask = torch.ones(
                    attention_mask.size(0), past_len - attention_mask.shape[1], device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, extended_mask], dim=1)
                cache_params = None

        if attention_mask.shape[1] < past_len:
            # we have to update manually the attention mask if
            # we are in decoding without cache
            # and we don't have position_ids here
            # TODO but we should be able to use cache_position though at a later time
            extended_mask = torch.ones(
                attention_mask.size(0), past_len - attention_mask.shape[1], device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, extended_mask], dim=1)
        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[HybridMamba2Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, HybridMamba2CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        arch_parameters = self.sampler.sample_step(self.get_arch_parameters())
        arch_params_sampled_dict = self.assign_arch_parameters(arch_parameters)
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba2_outputs = self.backbone(
            input_ids=input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            arch_params=arch_params_sampled_dict
        )
        hidden_states = mamba2_outputs[0]

        logits = self.mixop.forward(
                hidden_states.to(self.lm_head.weight.dtype), arch_params_sampled_dict["hidden_size"], self.lm_head_list).float()
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba2_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return logits, loss
        

    
    def estimate_mfu(self, fwdbwd_per_iter, dt): 
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        ####### TODO: DOUBLE CHECK FOR MAMBA #####
        N = self.get_num_params()
        # L, H, Q, T = max(self.layer_list), max(self.num_head_list), max(self.hidden_size_list) * self.config.expand // max(self.num_head_list), self.config.chunk_size
        # flops_per_token = 6*N + 24*L*H*Q*T
        # flops_per_fwdbwd = flops_per_token * T
        # flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        ## For Mamba2 Blocks:
        flops_per_iter = ((6 * N + 9 * self.config.state_size * max(self.hidden_size_list)*max(self.layer_list))  * self.config.chunk_size) * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 15.62e12 # 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.backbone.embeddings.weight.numel()
        return n_params
    
    def _init_arch_parameters(self):
        self.arch_parameter_dict = {}
        self.arch_num_hidden_layers = nn.Parameter(
            1e-3 * torch.randn([len(self.config.num_hidden_layers_list)]))
        self.arch_parameter_dict["num_hidden_layers"] = self.arch_num_hidden_layers
        self.arch_hidden_size = nn.Parameter(
            1e-3 * torch.randn([len(self.config.hidden_size_list)]))
        self.arch_parameter_dict["hidden_size"] = self.arch_hidden_size
        self.arch_num_heads = nn.Parameter(
            1e-3 * torch.randn([max(self.config.num_hidden_layers_list),len(self.config.num_heads_list)]))
        self.arch_parameter_dict["num_heads"] = self.arch_num_heads
        
    def assign_arch_parameters(self, arch_parameters):
        arch_params_dummy = {}
        for i, k in enumerate(self.arch_parameter_dict.keys()):
            arch_params_dummy[k] = arch_parameters[i]
        return arch_params_dummy
    
    def get_arch_parameters(self):
        return [self.arch_num_hidden_layers, self.arch_hidden_size, self.arch_num_heads]

    def get_model_parameters(self):
        return list(set(self.parameters()) - set(self.get_arch_parameters()))
    
    def get_best_config(self):
        #print(f"arch parameter {k}: {torch.nn.functional.softmax(model.module.arch_parameter_dict[k], dim=-1)}")
        best_config = {}
        #for k in self.arch_parameter_dict.keys():
        best_config["num_hidden_layers"] = self.layer_list[torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["num_hidden_layers"], dim=-1)).item()]
        best_config["hidden_size"] = self.hidden_size_list[torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["hidden_size"], dim=-1)).item()]
        best_num_heads = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["num_heads"], dim=-1),dim=-1)
        #print(best_num_heads)
        best_config["num_heads"] = [self.num_head_list[best_num_heads[i]] for i in range(best_num_heads.shape[0])]
        return best_config