"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from mixed_operations.mixed_embedding_attn import MixedEmbeddingAttention
from mixed_operations.mixed_attn_head_embed import MixedAttnHeadEmbed,  MixedAttnEmbd, MixedAttnHead
from mixed_operations.mixed_embedding import MixedEmbeddingV2
from mixed_operations.mixed_linear_head import MixedLinearHeadV2
from mixed_operations.mixed_linear_emb import MixedLinearV2Emb, MixedLinear
from mixed_operations.mixed_linear_emb_mlp import MixedLinearV2
from TangleNAS.search_spaces.tinyllama.mixed_operations.mixed_rms_norm import MixedLayerNormV2
from optimizers.optim_factory import get_mixop, get_sampler
from optimizers.mixop.entangle import EntangledOp
import itertools
# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = 1e-5

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class CausalSelfAttention(nn.Module):

    def __init__(self, config, mixop):
        super().__init__()
        self.embed_dim_list = config["embed_dim"]
        self.max_embed_dim = max(self.embed_dim_list)
        self.n_head_list = config["num_heads"]
        self.max_n_head = max(self.n_head_list)
        self.bias_att = config["bias"]
        self.dropout = config["dropout"]
        self.B = config["batch_size"]
        self.T = config["block_size"]
        self.C = self.max_embed_dim
        self.mixop = mixop
        self.flash = False #hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                                        .view(1, 1, config["block_size"], config["block_size"]))
        else:
            self.register_buffer("bias", None)
        assert self.max_embed_dim % self.max_n_head == 0
        # key, query, value projections for all heads, but in a batch
        c_attn = nn.Linear(self.max_embed_dim, 3 * self.max_embed_dim, bias=self.bias_att)
        self.c_attn_op = MixedEmbeddingAttention(self.embed_dim_list, self.max_embed_dim,c_attn)
        self.c_attn_op_list = self.get_entangle_ops(self.c_attn_op, self.embed_dim_list, "c_attn")
        # output projection
        c_proj = nn.Linear(self.max_embed_dim, self.max_embed_dim, bias=self.bias_att)
        self.c_proj_mix_op = MixedLinearV2Emb(config["embed_dim"], self.max_embed_dim, linear_layer=c_proj)
        self.c_proj_mix_op_list = self.get_entangle_ops(self.c_proj_mix_op, self.embed_dim_list, "c_proj_msa")
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.attention_op = MixedAttnHeadEmbed(self.n_head_list, self.embed_dim_list, self.dropout, self.B, self.T, self.C, self.attn_dropout, self.flash, self.bias)
        self.attention_op_list = self.get_entangle_ops_combi(self.attention_op, self.n_head_list, self.embed_dim_list, "attention_op")
        
   
    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def forward(self, x, i, arch_params):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        out  = self.mixop.forward(x, arch_params["embed_dim"], self.c_attn_op_list)
        y = self.mixop.forward(out, [arch_params["num_heads"][i],arch_params["embed_dim"]], self.attention_op_list, combi=True)
        y = self.mixop.forward(y, arch_params["embed_dim"], self.c_proj_mix_op_list)
        y = self.resid_dropout(y)
        return y

class MLP(nn.Module):

    def __init__(self, config, mixop):
        super().__init__()
        self.embed_dim_list = config["embed_dim"]
        self.max_embed_dim = max(self.embed_dim_list)
        self.bias = config["bias"]
        self.dropout = config["dropout"]
        self.max_mlp_ratio = max(config["mlp_ratio"])
        self.mixop = mixop
        c_fc    = nn.Linear(self.max_embed_dim, self.max_mlp_ratio * self.max_embed_dim, bias=self.bias)
        self.linear_0_mix_op = MixedLinearV2(config["embed_dim"], config["mlp_ratio"], linear_layer=c_fc)
        self.linear_0_mixed = self.get_entangle_ops_combi(self.linear_0_mix_op, config["embed_dim"], config["mlp_ratio"], "linear_embed_dim_c_fc")
        c_proj  = nn.Linear(self.max_embed_dim * self.max_mlp_ratio, self.max_embed_dim, bias=self.bias)
        self.linear_1_mix_op = MixedLinearV2(config["embed_dim"], config["mlp_ratio"], linear_layer=c_proj, reverse=True)
        self.linear_1_mixed = self.get_entangle_ops_combi(self.linear_1_mix_op, config["embed_dim"], config["mlp_ratio"], "linear_embed_dim_c_proj")
        self.dropout = nn.Dropout(self.dropout)

    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def forward(self, x, i , arch_params):
        x = self.mixop.forward(x, [arch_params["embed_dim"], arch_params["mlp_ratio"][i]], self.linear_0_mixed, combi=True)
        x = new_gelu(x)
        x = self.mixop.forward(x, [arch_params["embed_dim"], arch_params["mlp_ratio"][i]], self.linear_1_mixed, combi=True)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, mixop):
        super().__init__()
        self.embed_list = config["embed_dim"]
        self.max_embed_dim = max(self.embed_list)
        self.bias = config["bias"]
        self.mixop = mixop
        ln_1 = LayerNorm(self.max_embed_dim, bias=self.bias)
        self.ln1_op = MixedLayerNormV2(config["embed_dim"], self.max_embed_dim, ln_1)
        self.ln1_list = self.get_entangle_ops(self.ln1_op, config["embed_dim"], "ln1_block_"+str(id))
        self.attn = CausalSelfAttention(config, mixop)
        ln_2 = LayerNorm(self.max_embed_dim, bias=self.bias)
        self.ln2_op = MixedLayerNormV2(config["embed_dim"], self.max_embed_dim, ln_2)
        self.ln2_list = self.get_entangle_ops(self.ln2_op, config["embed_dim"], "ln2_block_"+str(id))
        self.mlp = MLP(config, mixop)

    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]
    
    def forward(self, x, i, arch_params=None):
        x = x + self.attn(self.mixop.forward(x, 
                        arch_params["embed_dim"], self.ln1_list), i, arch_params=arch_params)
        x = x + self.mlp(self.mixop.forward(x,
                        arch_params["embed_dim"], self.ln2_list), i, arch_params=arch_params)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config, use_we_v2=False):
        super().__init__()
        #print(config)
        assert config["vocab_size"] is not None
        assert config["block_size"] is not None
        self.config = config
        self.embed_list = config["embed_dim"]
        self.max_embed_dim = max(self.embed_list)
        self.layer_list = config["num_layers"]
        self.max_layer = max(self.layer_list)
        self.num_head = config["num_heads"]
        self.max_head = max(self.num_head)
        self.mlp_ratio = config["mlp_ratio"]
        self.max_mlp_ratio = max(self.mlp_ratio)
        self.vocab_size = config["vocab_size"]
        self.block_size = config["block_size"]
        self.dropout = config["dropout"]
        self.mixop = get_mixop(config["mixop"],use_we_v2=use_we_v2)
        self.sampler = get_sampler(config["mixop"])

        transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.max_embed_dim),
            wpe = nn.Embedding(self.block_size, self.max_embed_dim),
            ln_f = LayerNorm(self.max_embed_dim, bias=config["bias"]),
        ))
        self.transformer_h = nn.ModuleList([Block(config, self.mixop) for _ in range(self.max_layer)])
        self.transformer_drop = nn.Dropout(self.dropout)
        lm_head = nn.Linear(self.max_embed_dim, self.vocab_size, bias=False)
        transformer.wte.weight = lm_head.weight
        self.token_embedding_table_op = MixedEmbeddingV2(config["embed_dim"], max_embed_dim=self.max_embed_dim, embedding=transformer.wte)
        self.token_embedding_table_list = self.get_entangle_ops(self.token_embedding_table_op, config["embed_dim"], "embedding_table")
        self.position_embedding_table_op = MixedEmbeddingV2(config["embed_dim"], max_embed_dim=self.max_embed_dim, embedding=transformer.wpe)
        self.position_embedding_table_list = self.get_entangle_ops(self.position_embedding_table_op, config["embed_dim"], "position_embedding_table")
        self.ln_f_op = MixedLayerNormV2(config["embed_dim"], self.max_embed_dim, transformer.ln_f)
        self.ln_f_list = self.get_entangle_ops(self.ln_f_op, config["embed_dim"], "ln_f")
        self.lm_head_op = MixedLinearHeadV2(config["embed_dim"], self.max_embed_dim, lm_head)
        self.lm_head_list = self.get_entangle_ops(self.lm_head_op, config["embed_dim"], "lm_head")
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
         # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # CHECK THIS
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.max_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        self._init_arch_parameters()

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #for n,p in self.named_parameters():
        #    print(n,p.numel()/1e6)
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_arch_parameters(self):
        self.arch_parameter_dict = {}
        self.arch_num_layers = nn.Parameter(
            1e-3 * torch.randn([len(self.config["num_layers"])]))
        self.arch_parameter_dict["num_layers"] = self.arch_num_layers
        self.arch_embed_dim = nn.Parameter(
            1e-3 * torch.randn([len(self.config["embed_dim"])]))
        self.arch_parameter_dict["embed_dim"] = self.arch_embed_dim
        self.arch_num_heads = nn.Parameter(
            1e-3 * torch.randn([max(self.config["num_layers"]),len(self.config["num_heads"])]))
        self.arch_parameter_dict["num_heads"] = self.arch_num_heads
        self.arch_mlp_ratio = nn.Parameter(
            1e-3 * torch.randn([max(self.config["num_layers"]),len(self.config["mlp_ratio"])]))
        self.arch_parameter_dict["mlp_ratio"] = self.arch_mlp_ratio
        
    def assign_arch_parameters(self, arch_parameters):
        arch_params_dummy = {}
        for i, k in enumerate(self.arch_parameter_dict.keys()):
            arch_params_dummy[k] = arch_parameters[i]
        return arch_params_dummy
    
    def forward(self, idx, targets=None):
        arch_parameters = self.sampler.sample_step(self.get_arch_parameters())
        arch_params_sampled_dict = self.assign_arch_parameters(arch_parameters)
        device = idx.device
        b, t = idx.size()
        assert t <= self.config["block_size"], f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        tok_emb = self.mixop.forward(
            idx, arch_params_sampled_dict["embed_dim"], self.token_embedding_table_list)
        pos_emb = self.mixop.forward(
        pos, arch_params_sampled_dict["embed_dim"], self.position_embedding_table_list)
        x = self.transformer_drop(tok_emb + pos_emb)
        i = 0
        depth_output_list = []
        for block in self.transformer_h:
            x = block(x, i, arch_params_sampled_dict)
            if i+1 in self.config["num_layers"]:
                depth_output_list.append(x)
            i += 1
        x = self.mixop.forward_depth(
            depth_output_list, arch_params_sampled_dict["num_layers"])
        #x = self.transformer.ln_f(x)
        x = self.mixop.forward(
            x, arch_params_sampled_dict["embed_dim"], self.ln_f_list)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # logits = self.lm_head(x)
            logits = self.mixop.forward(
                x, arch_params_sampled_dict["embed_dim"], self.lm_head_list)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.mixop.forward(
                x[:, [-1], :], arch_params_sampled_dict["embed_dim"], self.lm_head_list)
            loss = None

        return logits, loss
    
    def get_best_config(self):
        #print(f"arch parameter {k}: {torch.nn.functional.softmax(model.module.arch_parameter_dict[k], dim=-1)}")
        best_config = {}
        #for k in self.arch_parameter_dict.keys():
        best_config["num_layers"] = self.layer_list[torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["num_layers"], dim=-1)).item()]
        best_config["embed_dim"] = self.embed_list[torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["embed_dim"], dim=-1)).item()]
        best_num_heads = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["num_heads"], dim=-1),dim=-1)
        #print(best_num_heads)
        best_config["num_heads"] = [self.num_head[best_num_heads[i]] for i in range(best_num_heads.shape[0])]
        best_mlp_ratio = torch.argmax(torch.nn.functional.softmax(self.arch_parameter_dict["mlp_ratio"], dim=-1),dim=-1)
        #print(best_mlp_ratio)
        best_config["mlp_ratio"] = [self.mlp_ratio[best_mlp_ratio[i]] for i in range(best_mlp_ratio.shape[0])]
        return best_config
    
    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
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

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head_op.linear_layer.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if "arch" not in pn}
        for pn in param_dict.keys():
            print(pn)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        print(f"decay: {decay}")
        print(f"no_decay: {no_decay}")
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = max(cfg["num_layers"]), max(cfg["num_heads"]), max(cfg["embed_dim"])//max(cfg["num_heads"]) , cfg["block_size"]
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_arch_parameters(self):
        return [self.arch_num_layers, self.arch_embed_dim, self.arch_num_heads,self.arch_mlp_ratio]
    
    def get_model_parameters(self):
        return list(set(self.parameters()) - set(self.get_arch_parameters()))
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# test the model
'''if __name__ == '__main__':
    config = {}
    config['num_layers'] = [1,2,3]
    config['num_heads'] = [1,2,3]
    config['embed_dim'] = [18,24,36]
    config["mlp_ratio"] = [1,2,4]
    config['vocab_size'] = 100
    config['block_size'] = 10
    config['bias'] = False
    config['dropout'] = 0.0
    config['batch_size'] = 2
    config['mixop'] = "gdas"
    model = GPT(config)
    model.sampler.set_taus(0.1,10)
    model.sampler.set_total_epochs(100)
    model.sampler.before_epoch()
    input = torch.randint(0, 100, (2, 10))
    target = torch.randint(0, 100, (2, 10))
    output = model(input, target)
    print(output)'''