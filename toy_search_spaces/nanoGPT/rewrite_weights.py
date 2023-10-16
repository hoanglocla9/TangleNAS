import torch
import torch.nn.functional as F
import itertools
def preprocess_combi(weights1, weights2):
    weights1 = preprocess_weights(weights1)
    weights2 = preprocess_weights(weights2)
    weights = weights1.reshape(weights1.shape[0], 1) @ weights2.reshape(
            1, weights2.shape[0])    
    return weights.flatten()

def preprocess_weights(weights):
    return weights
    
def compute_attn_op_mixture(weight, alpha_embed_dim, embed_choices):
    max_emb = max(embed_choices)
    weight_mix = 0
    for i in range(len(embed_choices)):
        embed_dim = embed_choices[i]
        weight_curr = alpha_embed_dim[i]*weight[:embed_dim*3,:embed_dim]
        weight_curr = F.pad(weight_curr,(0,max_emb-embed_dim,0,(3*max_emb)-(3*embed_dim)),'constant',0)
        weight_mix = weight_mix+weight_curr
    return weight_mix

def compute_embed_mixture(weight, alpha_embed_dim, embed_choices):
    weight_mix = 0
    max_emb = max(embed_choices)
    for i in range(len(embed_choices)):
        emb_choice = embed_choices[i]
        weight_curr = alpha_embed_dim[i]*weight[:,:emb_choice]
        weight_curr = F.pad(weight_curr,(0,max_emb-emb_choice,0,0),"constant",0)
        weight_mix += weight_curr
    return weight_mix

def compute_conv_mixture_proj(weight, bias, alpha_embed_dim, embed_choices):
    weights_conv_mix = 0
    bias_conv_mix = 0
    max_emb = max(embed_choices)
    #print(embed_choices)
    for i in range(len(embed_choices)):
        emb_choice = embed_choices[i]
        weight_curr = alpha_embed_dim[i] * weight[:emb_choice,:,:,:]
        weight_curr = F.pad(weight_curr, (0,0,0,0,0,0,0,max_emb-emb_choice), "constant", 0)
        bias_curr = alpha_embed_dim[i] * bias[:emb_choice]
        bias_curr = F.pad(bias_curr,(0,max_emb-emb_choice),"constant",0)
        weights_conv_mix += weight_curr
        bias_conv_mix += bias_curr
    return weights_conv_mix, bias_conv_mix

def compute_qkv_mixture(weight, bias, alphas, embed_choices, head_choices):
    emb_head_choice_list = list(itertools.product(embed_choices, head_choices))
    out_dim_max = max(head_choices) * 64 * 3
    max_emb = max(embed_choices)
    weight_mix = 0
    bias_mix = 0
    print(alphas)
    for i in range(len(emb_head_choice_list)):
        emb_size, head_choice = emb_head_choice_list[i]
        out_dim = head_choice*64*3
        alpha = alphas[i]
        weight_curr = alpha*weight[:,:emb_size]
        weight_curr = torch.cat([weight_curr[i:out_dim:3, :] for i in range(3)], dim=0)
        weight_curr = F.pad(weight_curr,(0,max_emb-emb_size,0,out_dim_max-out_dim))
        weight_mix += weight_curr
        bias_curr = alpha*bias[:out_dim]
        bias_curr = F.pad(bias_curr, (0,out_dim_max-out_dim), "constant", 0)
        bias_mix += bias_curr
    return weight_mix, bias_mix

def compute_attn_proj_weights(weight, alphas, embed_choices):
    max_emb = max(embed_choices)
    weight_mix = 0
    for i in range(len(embed_choices)):
        emb = embed_choices[i]
        weight_curr = alphas[i]*weight[:emb,:emb]
        weight_curr = F.pad(weight_curr,(0,max_emb-emb,0,max_emb-emb),'constant',0)
        weight_mix = weight_mix + weight_curr
    return weight_mix


def compute_layer_norm_mix_weights_bias(weights, bias, alphas, embed_choices):
    weight_mix = 0
    bias_mix = 0
    max_emb = max(embed_choices)
    for i in range(len(embed_choices)):
        weight_curr = alphas[i]*weights[:embed_choices[i]]
        bias_curr = alphas[i] * bias[:embed_choices[i]]
        weight_curr = F.pad(weight_curr, (0, max_emb-embed_choices[i]), "constant", 0)
        bias_curr = F.pad(bias_curr, (0, max_emb-embed_choices[i]), "constant", 0)
        weight_mix += weight_curr
        bias_mix += bias_curr
    return weight_mix, bias_mix

def compute_layer_norm_mix_weights(weights, alphas, embed_choices):
    weight_mix = 0
    bias_mix = 0
    max_emb = max(embed_choices)
    for i in range(len(embed_choices)):
        weight_curr = alphas[i]*weights[:embed_choices[i]]
        weight_curr = F.pad(weight_curr, (0, max_emb-embed_choices[i]), "constant", 0)
        weight_mix += weight_curr
    return weight_mix

def compute_linear_emb_expand(weights, alphas, embed_choices, ratio_choices):
    weight_mix = 0
    emb_ratio_choice_list = list(itertools.product(embed_choices, ratio_choices))
    max_dim = max(embed_choices)*max(ratio_choices)
    max_emb = max(embed_choices)
    for i in range(len(emb_ratio_choice_list)):
        alpha = alphas[i]
        emb_size, ratio = emb_ratio_choice_list[i]
        weight_curr = alpha * weights[:int(emb_size * ratio
                                            ), :emb_size]
        #print(weight_curr.shape)
        weight_curr = F.pad(weight_curr, (0,max_emb-emb_size,0,max_dim-int(emb_size * ratio)), "constant", 0)
        #print(weight_curr.shape)
        weight_mix += weight_curr
    return weight_mix

def compute_linear_emb_contract(weights, alphas, embed_choices, ratio_choices):
    weight_mix = 0
    emb_ratio_choice_list = list(itertools.product(embed_choices, ratio_choices))
    max_dim = max(embed_choices)*max(ratio_choices)
    max_emb = max(embed_choices)
    for i in range(len(emb_ratio_choice_list)):
        alpha = alphas[i]
        emb_size, ratio = emb_ratio_choice_list[i]
        weight_curr = alpha * weights[:emb_size,:int(emb_size * ratio
                                            )]
        weight_curr = F.pad(weight_curr, (0,max_dim-int(emb_size * ratio),0,max_emb-emb_size), "constant", 0)
        weight_mix += weight_curr
    return weight_mix

def compute_linear_head_mixture(weights, alphas, embed_choices):
    weights_mix = 0
    max_emb = max(embed_choices)
    for i in range(len(embed_choices)):
        weights_curr = alphas[i]*weights[:,:embed_choices[i]]
        weights_curr= F.pad(weights_curr, (0, max_emb-embed_choices[i], 0, 0), "constant", 0)
        weights_mix += weights_curr
    return weights_mix

def recompute_weights(model_path, save_path):
    config = {
        "embed_dim": [18,24,36],
        "mlp_ratio": [1,2,4],
        "layer_num": [1,2,3],
        "num_heads": [1,2,3]
    }
    model_state_dict = torch.load(model_path, map_location="cpu")
    print(list(model_state_dict.keys()))
    new_state_dict = {}
    alpha_embed_dim = torch.nn.functional.softmax(model_state_dict["arch_embed_dim"],dim=-1)
    new_state_dict["arch_embed_dim"] = model_state_dict["arch_embed_dim"]
    alphas_mlp_ratio = torch.nn.functional.softmax(model_state_dict["arch_mlp_ratio"],dim=-1)
    new_state_dict["arch_mlp_ratio"] = model_state_dict["arch_mlp_ratio"]
    alphas_num_heads = torch.nn.functional.softmax(model_state_dict["arch_num_heads"],dim=-1)
    new_state_dict["arch_num_heads"] = model_state_dict["arch_num_heads"]
    alphas_layer_num = torch.nn.functional.softmax(model_state_dict["arch_num_layers"],dim=-1)
    new_state_dict["arch_num_layers"] = model_state_dict["arch_num_layers"]
    new_state_dict["token_embedding_table_op.embedding.weight"] = compute_embed_mixture(model_state_dict["token_embedding_table_op.embedding.weight"], alpha_embed_dim, config["embed_dim"])
    new_state_dict["position_embedding_table_op.embedding.weight"] = compute_embed_mixture(model_state_dict["position_embedding_table_op.embedding.weight"], alpha_embed_dim, config["embed_dim"])
    
    for i in range(0,max(config["layer_num"])):
        alphas_emb_ratio = preprocess_combi(alpha_embed_dim, alphas_mlp_ratio[i])
        layer_key = "transformer_h."+str(i)+".ln1_op.layer_norm.weight"
        new_state_dict[layer_key] = compute_layer_norm_mix_weights(model_state_dict[layer_key], alpha_embed_dim, config["embed_dim"])
        layer_key = "transformer_h."+str(i)+".ln2_op.layer_norm.weight"
        new_state_dict[layer_key] = compute_layer_norm_mix_weights(model_state_dict[layer_key], alpha_embed_dim, config["embed_dim"])
        layer_key = "transformer_h."+str(i)+".attn.c_proj_mix_op.linear_layer.weight"
        new_state_dict[layer_key] = compute_attn_proj_weights(model_state_dict[layer_key], alpha_embed_dim, config["embed_dim"])
        layer_key = "transformer_h."+str(i)+".attn.c_attn_op.linear_layer.weight"
        new_state_dict[layer_key] = compute_attn_op_mixture(model_state_dict[layer_key], alpha_embed_dim, config["embed_dim"])
        layer_key = "transformer_h."+str(i)+".mlp.linear_0_mix_op.linear_layer.weight"
        new_state_dict[layer_key] = compute_linear_emb_expand(model_state_dict[layer_key],alphas_emb_ratio , config["embed_dim"], config["mlp_ratio"])
        layer_key = "transformer_h."+str(i)+".mlp.linear_1_mix_op.linear_layer.weight"
        new_state_dict[layer_key] = compute_linear_emb_contract(model_state_dict[layer_key], alphas_emb_ratio, config["embed_dim"], config["mlp_ratio"])
    new_state_dict["lm_head_op.linear_layer.weight"] = compute_linear_head_mixture(model_state_dict["lm_head_op.linear_layer.weight"], alpha_embed_dim, config["embed_dim"])
    new_state_dict["ln_f_op.layer_norm.weight"] = compute_layer_norm_mix_weights(model_state_dict["ln_f_op.layer_norm.weight"], alpha_embed_dim, config["embed_dim"]) 
    torch.save(new_state_dict,save_path) 
    
recompute_weights("/gpfs/bwfor/work/ws/fr_rs1131-tanglenas/TangleNAS-dev/gpt_checkpoint.pth","rewrite_weights_gpt2.pth")
print(torch.load("rewrite_weights_gpt2.pth").keys())