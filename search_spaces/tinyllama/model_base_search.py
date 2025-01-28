# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from quantizer import Quantizer

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin

from optimizers.optim_factory import get_mixop, get_sampler
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
# from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import inspect

from transformers.models.llama.configuration_llama import LlamaConfig
from mixed_operations.mixed_rms_norm import MixedRMSNorm
from mixed_operations.mixed_mlp_linear import MixedLinear
from mixed_operations.mixed_attn_head_embed import MixedAttnHeadEmbed
from mixed_operations.mixed_attn_linear import MixedLinear_KV, MixedLinear_QO
from mixed_operations.mixed_embedding import MixedEmbedding
from mixed_operations.mixed_head_linear import MixedLinear_Head

from optimizers.mixop.entangle import EntangledOp
import itertools

logger = logging.get_logger(__name__)


def get_entangle_ops(op, choices, op_name):
    return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

def get_entangle_ops_combi(op, choices1, choices2, choices3=None, choices4=None, op_name=''):
    choices = list(itertools.product(choices1, choices2))
    if choices3 is not None:
        choices = list(itertools.product(choices, choices3))
    if choices4 is not None:
        choices = list(itertools.product(choices, choices4))

    return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.base = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.attention_scaling = 1.0 # default

    @torch.no_grad()
    def forward(self, x, head_dim, position_ids):
        ## x is value_states
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        dim = int(head_dim * self.partial_rotary_factor)
        # Compute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device_type) / dim))
        # Core RoPE block
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        if cos.size(-1) == dim + 1:
            cos = cos[:,:,:dim]
            sin = cos[:,:,:dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config, mixop, layer_idx):
        super().__init__()
        self.config = config
        self.mixop = mixop
        self.layer_idx = layer_idx

        self.a_bit_list = config.a_bit_list
        self.w_bit_list = config.w_bit_list

        self.hidden_size_list = config.hidden_size
        self.intermediate_size_list = config.intermediate_size

        gate_proj = nn.Linear(max(self.hidden_size_list), max(self.intermediate_size_list), bias=config.mlp_bias)
        self.gate_proj_op = MixedLinear(self.hidden_size_list,  self.intermediate_size_list, self.a_bit_list, self.w_bit_list, gate_proj, False)
        self.gate_proj_op_list = get_entangle_ops_combi(self.gate_proj_op, self.hidden_size_list, self.intermediate_size_list, self.a_bit_list, self.w_bit_list, "gate_proj")

        up_proj = nn.Linear(max(self.hidden_size_list), max(self.intermediate_size_list), bias=config.mlp_bias)
        self.up_proj_op = MixedLinear(self.hidden_size_list,  self.intermediate_size_list, self.a_bit_list, self.w_bit_list, up_proj, False)
        self.up_proj_op_list = get_entangle_ops_combi(self.up_proj_op, self.hidden_size_list, self.intermediate_size_list, self.a_bit_list, self.w_bit_list,  "gate_proj")

        down_proj = nn.Linear(max(self.intermediate_size_list), max(self.hidden_size_list), bias=config.mlp_bias)
        self.down_proj_op = MixedLinear(self.hidden_size_list,  self.intermediate_size_list, self.a_bit_list, self.w_bit_list, down_proj, True)
        self.down_proj_op_list = get_entangle_ops_combi(self.down_proj_op, self.intermediate_size_list, self.hidden_size_list, self.a_bit_list, self.w_bit_list, "down_proj")

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, arch_params):
        x_up = self.mixop.forward(x, 
                        [arch_params["hidden_size"], arch_params['intermediate_size'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]], #
                        self.up_proj_op_list, combi=True)
        x_gate = self.act_fn(self.mixop.forward(x, 
                        [arch_params["hidden_size"], arch_params['intermediate_size'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]], #
                        self.gate_proj_op_list, combi=True))
        x_gate_up = x_gate * x_up 
        x = self.mixop.forward(x_gate_up, 
                        [arch_params["hidden_size"], arch_params['intermediate_size'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]], #
                        self.down_proj_op_list, combi=True)
        return x



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, mixop=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.attention_bias = config.attention_bias
        self.flash = True if config._attn_implementation == "flash_attention_2" else False
        self.hidden_size_list = config.hidden_size
        self.num_heads_list = config.num_heads
        self.max_head_dim = max(self.hidden_size_list) // min(self.num_heads_list) ### 
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.mixop = mixop 
        self.a_bit_list = config.a_bit_list
        self.w_bit_list = config.w_bit_list

        q_proj = nn.Linear(max(self.hidden_size_list), min(self.num_heads_list) * self.max_head_dim, bias=config.attention_bias)
        self.q_proj_op = MixedLinear_QO(self.hidden_size_list, self.num_heads_list, self.a_bit_list, self.w_bit_list, q_proj)
        self.q_proj_op_list = get_entangle_ops_combi(self.q_proj_op, self.hidden_size_list, self.num_heads_list, self.a_bit_list, self.w_bit_list, "q_proj")

        k_proj = nn.Linear(max(self.hidden_size_list), self.num_key_value_heads * self.max_head_dim, bias=config.attention_bias)
        self.k_proj_op = MixedLinear_KV(self.hidden_size_list, self.num_heads_list, self.num_key_value_heads, self.a_bit_list, self.w_bit_list, k_proj)
        self.k_proj_op_list = get_entangle_ops_combi(self.k_proj_op, self.hidden_size_list, self.num_heads_list, self.a_bit_list, self.w_bit_list, "k_proj")

        v_proj = nn.Linear(max(self.hidden_size_list), self.num_key_value_heads * self.max_head_dim, bias=config.attention_bias)
        self.v_proj_op = MixedLinear_KV(self.hidden_size_list, self.num_heads_list, self.num_key_value_heads, self.a_bit_list, self.w_bit_list, v_proj)
        self.v_proj_op_list = get_entangle_ops_combi(self.v_proj_op, self.hidden_size_list, self.num_heads_list, self.a_bit_list, self.w_bit_list, "v_proj")

        o_proj = nn.Linear(min(self.num_heads_list) * self.max_head_dim, max(self.hidden_size_list), bias=config.attention_bias)
        self.o_proj_op = MixedLinear_QO(self.hidden_size_list, self.num_heads_list, self.a_bit_list, self.w_bit_list, o_proj, reverse=True)
        self.o_proj_op_list = get_entangle_ops_combi(self.o_proj_op, self.hidden_size_list, self.num_heads_list, self.a_bit_list, self.w_bit_list, "o_proj")
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.attn_dropout = nn.Dropout(self.attention_dropout) ## check
        self.attention_op = MixedAttnHeadEmbed(self.num_heads_list, self.hidden_size_list, self.num_key_value_heads, 
                                                self.attention_dropout, self.attn_dropout, self.rotary_emb, 
                                                self.flash, self.attention_bias)
        self.attention_op_list = get_entangle_ops_combi(self.attention_op, self.num_heads_list, self.hidden_size_list, op_name="attention_op")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        arch_params:dict = {},
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.mixop.forward(hidden_states, 
                               [arch_params["hidden_size"], arch_params['num_heads'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]], #
                                self.q_proj_op_list, combi=True)
        k = self.mixop.forward(hidden_states, 
                               [arch_params["hidden_size"], arch_params['num_heads'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]],  # 
                               self.k_proj_op_list, combi=True)
        v = self.mixop.forward(hidden_states, 
                               [arch_params["hidden_size"], arch_params['num_heads'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]],  #
                               self.v_proj_op_list, combi=True)


        x = q, k, v, position_ids, position_embeddings, attention_mask
        
        attn_output = self.mixop.forward(x, [arch_params["hidden_size"], arch_params['num_heads'][self.layer_idx]], self.attention_op_list, combi=True)
        attn_output = self.mixop.forward(attn_output, 
                            [arch_params["hidden_size"], arch_params['num_heads'][self.layer_idx], arch_params['a_bit'][self.layer_idx], arch_params['w_bit'][self.layer_idx]], #
                            self.o_proj_op_list, combi=True) 
        
        return attn_output, None, past_key_value



class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, mixop=None):
        super().__init__()
        self.hidden_size_list = config.hidden_size
        self.mixop = mixop
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, mixop=mixop)

        self.mlp = LlamaMLP(config, layer_idx=layer_idx, mixop=mixop)

        input_layernorm = LlamaRMSNorm(max(self.hidden_size_list), eps=config.rms_norm_eps)
        self.input_layernorm_op = MixedRMSNorm(self.hidden_size_list, input_layernorm)
        self.input_layernorm_op_list = get_entangle_ops(self.input_layernorm_op, self.hidden_size_list, "input_layernorm")

        post_attention_layernorm = LlamaRMSNorm(max(self.hidden_size_list), eps=config.rms_norm_eps)
        self.post_attention_layernorm_op = MixedRMSNorm(self.hidden_size_list, post_attention_layernorm)
        self.post_attention_layernorm_op_list = get_entangle_ops(self.post_attention_layernorm_op, self.hidden_size_list, "post_attention_layernorm")


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        arch_params: dict = {},
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        residual = hidden_states

        hidden_states = self.mixop.forward(hidden_states, arch_params["hidden_size"], self.input_layernorm_op_list)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
            arch_params=arch_params,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mixop.forward(hidden_states, arch_params["hidden_size"], self.post_attention_layernorm_op_list)
        hidden_states = self.mlp(hidden_states, arch_params=arch_params)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, mixop=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.mixop = mixop
        self.hidden_size_list = config.hidden_size
        self.num_hidden_layers_list = config.num_hidden_layers

        embed_tokens = nn.Embedding(config.vocab_size, max(self.hidden_size_list), self.padding_idx)
        self.embed_tokens_op = MixedEmbedding(self.hidden_size_list, embed_tokens)
        self.embed_tokens_op_list = get_entangle_ops(self.embed_tokens_op, self.hidden_size_list, "embed_tokens")

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx, mixop=mixop) for layer_idx in range(max(self.num_hidden_layers_list))]
        )

        norm = LlamaRMSNorm(max(self.hidden_size_list), eps=config.rms_norm_eps)
        self.norm_op = MixedRMSNorm(self.hidden_size_list, norm)
        self.norm_op_list = get_entangle_ops(self.norm_op, self.hidden_size_list, "norm_op")

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        arch_params: dict = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


        inputs_embeds = self.mixop.forward(input_ids, arch_params['hidden_size'], self.embed_tokens_op_list)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = None # self.rotary_emb(hidden_states, ,position_ids)

        # decoder layers
        depth_output_list = []
        i = 0
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=False,
                position_embeddings=position_embeddings,
                arch_params=arch_params
            )

            hidden_states = layer_outputs[0]
            if i+1 in self.config.num_hidden_layers:
                depth_output_list.append(hidden_states)
            i += 1

        hidden_states = self.mixop.forward_depth(
            depth_output_list, arch_params["num_hidden_layers"])
        
        hidden_states = self.mixop.forward(hidden_states, arch_params['hidden_size'], self.norm_op_list)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache :
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and attention_mask is not None
        #     and attention_mask.device.type == "cuda"
        # ):
        #     # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        #     # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        #     # Details: https://github.com/pytorch/pytorch/issues/110213
        #     min_dtype = torch.finfo(dtype).min
        #     causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mixop = get_mixop(config.mixop,use_we_v2=True)
        self.sampler = get_sampler(config.mixop)
        self.hidden_size_list = config.hidden_size

        self.model = LlamaModel(config, mixop=self.mixop)
        self.vocab_size = config.vocab_size
        lm_head = nn.Linear(max(self.hidden_size_list), config.vocab_size, bias=False)
        self.lm_head_op = MixedLinear_Head(self.hidden_size_list, self.vocab_size, linear_layer=lm_head)
        self.lm_head_op_list = get_entangle_ops(self.lm_head_op, self.hidden_size_list, "lm_head")
        # Initialize weights and apply final processing
        self.post_init()
        self._init_arch_parameters()

    def get_input_embeddings(self):
        return self.model.embed_tokens_op_list[-1].op

    def set_input_embeddings(self, value):
        self.model.embed_tokens_op_list[-1].op.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head_op_list[-1].op.linear_layer

    def set_output_embeddings(self, new_embeddings):
        self.lm_head_op_list[-1].op.linear_layer = new_embeddings


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if torch.isnan(self.get_arch_parameters()[-1]).any():
            print(self.get_arch_parameters()[0])
            exit()
        arch_parameters = self.sampler.sample_step(self.get_arch_parameters())
        arch_params_sampled_dict = self.assign_arch_parameters(arch_parameters)
        device = input_ids.device

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            arch_params=arch_params_sampled_dict
        )

        hidden_states = outputs[0]
        # 
        logits = self.mixop.forward(hidden_states[:, -num_logits_to_keep:, :], arch_params_sampled_dict["hidden_size"], self.lm_head_op_list)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.embed_tokens.weight.numel()
        return n_params

    def get_best_config(self):
        best_config = {}
        best_config["num_hidden_layers"] = self.config.num_hidden_layers[torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["num_hidden_layers"], dim=-1)).item()]
        best_config["hidden_size"] = self.config.hidden_size[torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["hidden_size"], dim=-1)).item()]

        best_num_heads = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["num_heads"], dim=-1),dim=-1)
        best_config["num_heads"] = [self.config.num_heads[best_num_heads[i]] for i in range(best_num_heads.shape[0])]

        best_intermediate_size = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["intermediate_size"], dim=-1),dim=-1)
        best_config["intermediate_size"] = [self.config.intermediate_size[best_intermediate_size[i]] for i in range(best_intermediate_size.shape[0])]

        best_a_bit = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["a_bit"], dim=-1),dim=-1)
        best_config["a_bit"] = [self.config.a_bit_list[best_a_bit[i]] for i in range(best_a_bit.shape[0])]

        best_w_bit = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["w_bit"], dim=-1),dim=-1)
        best_config["w_bit"] = [self.config.w_bit_list[best_w_bit[i]] for i in range(best_w_bit.shape[0])]

        return best_config
    
    def _init_arch_parameters(self):
        self.arch_parameter_dict = {}
        self.arch_num_hidden_layers = nn.Parameter(
            1e-3 * torch.randn([len(self.config.num_hidden_layers)]))
        self.arch_parameter_dict["num_hidden_layers"] = self.arch_num_hidden_layers

        self.arch_hidden_size = nn.Parameter(
            1e-3 * torch.randn([len(self.config.hidden_size)]))
        self.arch_parameter_dict["hidden_size"] = self.arch_hidden_size

        self.arch_num_heads = nn.Parameter(
            1e-3 * torch.randn([max(self.config.num_hidden_layers),len(self.config.num_heads)]))
        self.arch_parameter_dict["num_heads"] = self.arch_num_heads

        self.arch_intermediate_size = nn.Parameter(
            1e-3 * torch.randn([max(self.config.num_hidden_layers),len(self.config.intermediate_size)]))
        self.arch_parameter_dict["intermediate_size"] = self.arch_intermediate_size

        if self.config.a_bit_list is not None:
            self.arch_a_bit = nn.Parameter(
                1e-3 * torch.randn([max(self.config.num_hidden_layers),len(self.config.a_bit_list)]))
            self.arch_parameter_dict["a_bit"] = self.arch_a_bit

            self.arch_w_bit= nn.Parameter(
                1e-3 * torch.randn([max(self.config.num_hidden_layers),len(self.config.w_bit_list)]))
            self.arch_parameter_dict["w_bit"] = self.arch_w_bit
        
    def assign_arch_parameters(self, arch_parameters):
        arch_params_dummy = {}
        for i, k in enumerate(self.arch_parameter_dict.keys()):
            arch_params_dummy[k] = arch_parameters[i]
        return arch_params_dummy
    
    def get_arch_parameters(self):
        if self.config.a_bit_list is not None:
            return [self.arch_num_hidden_layers, self.arch_hidden_size, self.arch_num_heads, self.arch_intermediate_size, self.arch_a_bit, self.arch_w_bit]
        else:
            return [self.arch_num_hidden_layers, self.arch_hidden_size, self.arch_num_heads, self.arch_intermediate_size] # , self.arch_a_bit, self.arch_w_bit
        
    def get_model_parameters(self):
        return list(set(self.parameters()) - set(self.get_arch_parameters()) - set(self.get_quantization_parameters())) #
    
    def get_quantization_parameters(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if "arch" not in pn}
        quant_params = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('a_scale') or pn.endswith('w_scale'):
                    quant_params.add(fpn)
        return [param_dict[pn] for pn in sorted(list(quant_params))]
    
    def verify_size(self, fixed_config_file):
        import json
        with open(fixed_config_file, "r") as f:
            fixed_config = json.load(f)
        assert fixed_config["hidden_size"] == max(self.config.hidden_size) and  \
                fixed_config["intermediate_size"] == max(self.config.intermediate_size) and \
                fixed_config["num_attention_heads"] == min(self.config.num_heads)
                # fixed_config["num_hidden_layers"] == max(self.config.num_hidden_layers) and \

    def is_flexible_params(self, param_name):
        flexible_params = ['lm_head', 'embed_tokens', 'norm', 'input_layernorm', 'post_attention_layernorm', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention',
                           'gate_proj', 'up_proj', 'down_proj']
        for fpn in flexible_params:
            if fpn in param_name:
                return fpn
        return None
    

    def from_pretrained(self, model_path):
        import json, os
        from safetensors.torch import load_file

        fixed_config_file = os.path.join(model_path, "config.json")
        self.verify_size(fixed_config_file)
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        elif os.path.exists(os.path.join(model_path, "model.safetensors")):
            weight_path = os.path.join(model_path, "model.safetensors")
            state_dict = load_file(weight_path)
        else:
            raise 'Exception not exists wieght file in directory'

        # for fpn in self.state_dict():
        #     if 'layers' not in fpn:
        #         print(fpn)
                
        for p_n in state_dict.keys():
            if p_n not in self.state_dict():
                extracted_param_name = self.is_flexible_params(p_n)
                if extracted_param_name is not None:
                    flexible_param_name = p_n.replace(extracted_param_name, extracted_param_name+"_op")
                    if "lm_head_op" in flexible_param_name:
                        flexible_param_name = "lm_head_op.linear_layer.weight"

                    if flexible_param_name not in self.state_dict():
                        # raise 'Missing this params ' + p_n
                        pass
                    else:
                        self.state_dict()[flexible_param_name].copy_(state_dict[p_n])

                else:
                    raise 'Missing this params ' + p_n
            else:
                self.state_dict()[p_n].copy_(state_dict[p_n])
                
        del state_dict

        # # init a huggingface/transformers model
        # model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # sd_hf = model_hf.state_dict()

        # # copy while ensuring all of the parameters are aligned and match in names and shapes
        # sd_keys_hf = sd_hf.keys()
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # for k in sd_keys_hf:
        #     if any(k.endswith(w) for w in transposed):
        #         # special treatment for the Conv1D weights we need to transpose
        #         assert sd_hf[k].shape[::-1] == sd[k].shape
        #         with torch.no_grad():
        #             sd[k].copy_(sd_hf[k].t())
        #     else:
        #         # vanilla copy over the other parameters
        #         assert sd_hf[k].shape == sd[k].shape
        #         with torch.no_grad():
        #             sd[k].copy_(sd_hf[k])


    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        quant_params = set()
        whitelist_weight_modules = (MixedLinear, MixedLinear_Head, MixedLinear_KV, MixedLinear_QO)
        blacklist_weight_modules = (LlamaRMSNorm, LlamaRotaryEmbedding, torch.nn.Embedding, MixedRMSNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
              if "arch" not in pn:
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('a_scale') or pn.endswith('w_scale'):
                    quant_params.add(fpn)
        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        # decay.remove('lm_head_op.linear_layer.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if "arch" not in pn}
        # for pn in param_dict.keys():
        #     print(pn)
        inter_params = decay & no_decay & quant_params
        union_params = decay | no_decay | quant_params
        # print(union_params)
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay/quant_params sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay/quant_params set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        # print(f"decay: {decay}")
        # print(f"no_decay: {no_decay}")
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


        q_optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(quant_params))], "weight_decay": 0.0},
        ]
        optimizer_q = torch.optim.AdamW(q_optim_groups, lr=learning_rate, **extra_args) #

        return optimizer, optimizer_q
    
    def auxiliary_quantized_loss(self, fairness_regularization=False, quantization_error_minimization=False):
        QE_loss, distribution_loss = 0, 0

        for _, module in self.named_modules():
            if isinstance(module, (MixedLinear, MixedLinear_Head, MixedLinear_KV, MixedLinear_QO)):
                quantizer = module.quantizer

                if isinstance(quantizer, Quantizer):
                    if module.w_bit_list is None:
                        continue
                    weights = module.weight
                    QE_loss_per_layer = 0
                    for current_wbits in module.w_bit_list:
                        step_size = quantizer.get_wscale(current_wbits, detach=False)
                        
                        is_computed_clipped_weights = False
                        if fairness_regularization: # here we only force the distribution within the highly decoupled subsets...
                            if current_wbits == 2:
                                lower_bound, upper_bound = quantizer.weight_bound(bits=current_wbits)
                                clipped_weights = torch.clamp(weights, min=lower_bound, max=upper_bound)
                                distribution_loss += 1/2 * clipped_weights.pow(2).sum() # must using SGD for weight quantization

                                is_computed_clipped_weights = True

                        if quantization_error_minimization:
                            if not is_computed_clipped_weights:
                                lower_bound, upper_bound = quantizer.weight_bound(bits=current_wbits)
                                clipped_weights = torch.clamp(weights, min=lower_bound, max=upper_bound)

                            _, q_weights = quantizer(None, weights, abits=None, wbits=current_wbits)
                            q_weights = q_weights.detach()
                            bit_wise_distance = 2**(current_wbits - min(module.w_bit_list))

                            if bit_wise_distance != 1:
                                step_size = step_size.detach()
                                thd_neg_min, thd_pos_min = quantizer.compute_thd(min(module.w_bit_list))
                                bit_wise_distance_mapping = [ele*bit_wise_distance*step_size for ele in range(thd_neg_min, thd_pos_min+1)]

                                idx = q_weights == bit_wise_distance_mapping[0]
                                for cod in bit_wise_distance_mapping[1:]:
                                    idx |= (q_weights == cod)
                                
                                latent_weights = clipped_weights.detach()
                                q_weights = torch.where(idx, q_weights, latent_weights)
                            
                            QE_loss_per_layer += ((clipped_weights - q_weights) ** 2).sum(0).mean()
                    QE_loss_per_layer /= len(module.w_bit_list)
                    QE_loss += QE_loss_per_layer

        return QE_loss, distribution_loss
    

    def auxiliary_quantized_loss_v2(self, fairness_regularization=False, quantization_error_minimization=False):
        QE_loss, distribution_loss = 0, 0

        for _, module in self.named_modules():
            if isinstance(module, (MixedLinear, MixedLinear_Head, MixedLinear_KV, MixedLinear_QO)):
                quantizer = module.quantizer

                if isinstance(quantizer, Quantizer):
                    if module.w_bit_list is None:
                        continue
                    weights = module.weight
                    QE_loss_per_layer = 0
                    for current_wbits in module.w_bit_list:
                        step_size = quantizer.get_wscale(current_wbits, detach=False)
                        
                        is_computed_clipped_weights = False
                        if fairness_regularization: # here we only force the distribution within the highly decoupled subsets...
                            if current_wbits == 2:
                                lower_bound, upper_bound = quantizer.weight_bound(bits=current_wbits)
                                clipped_weights = torch.clamp(weights, min=lower_bound, max=upper_bound)
                                distribution_loss += 1/2 * clipped_weights.pow(2).sum() # must using SGD for weight quantization

                                is_computed_clipped_weights = True

                        if quantization_error_minimization:
                            if not is_computed_clipped_weights:
                                lower_bound, upper_bound = quantizer.weight_bound(bits=current_wbits)
                                clipped_weights = torch.clamp(weights, min=lower_bound, max=upper_bound)

                            _, q_weights = quantizer(None, weights, abits=None, wbits=current_wbits)
                            q_weights = q_weights.detach()
                            bit_wise_distance = 2**(current_wbits - min(module.w_bit_list))

                            if bit_wise_distance != 1:
                                step_size = step_size.detach()
                                thd_neg_min, thd_pos_min = quantizer.compute_thd(min(module.w_bit_list))
                                bit_wise_distance_mapping = [ele*bit_wise_distance*step_size for ele in range(thd_neg_min, thd_pos_min+1)]

                                idx = q_weights == bit_wise_distance_mapping[0]
                                for cod in bit_wise_distance_mapping[1:]:
                                    idx |= (q_weights == cod)
                                
                                latent_weights = clipped_weights.detach()
                                q_weights = torch.where(idx, q_weights, latent_weights)
                            
                            QE_loss_per_layer += ((clipped_weights - q_weights) ** 2).sum(0).mean()
                    QE_loss_per_layer /= len(module.w_bit_list)
                    QE_loss += QE_loss_per_layer

        return QE_loss, distribution_loss