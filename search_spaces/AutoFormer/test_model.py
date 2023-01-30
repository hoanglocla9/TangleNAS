from model_autoformer.supernet_transformer_inherit_base import Vision_TransformerSuper, derive_best_config
import torch
import random


def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['layer_num'])
    for dimension in dimensions:
        config[dimension] = [
            random.choice(choices[dimension]) for _ in range(depth)
        ]

    config['embed_dim'] = [random.choice(choices['embed_dim'])] * depth

    config['layer_num'] = depth
    return config


if __name__ == "__main__":
    n_classes = 10
    img_size = 224
    patch_size = 16
    # Initialize the search space and model
    # TODO: add an argparser here at some point
    config = {
        "embed_dim": [60, 120],
        "mlp_ratio": [1, 2],
        "layer_num": [4, 6],
        "num_heads": [1, 2]
    }
    model = Vision_TransformerSuper(img_size=img_size,
                                    patch_size=patch_size,
                                    embed_dim=120,
                                    depth=6,
                                    num_heads=1,
                                    mlp_ratio=2,
                                    qkv_bias=True,
                                    drop_rate=0,
                                    drop_path_rate=0.1,
                                    gp=True,
                                    num_classes=n_classes,
                                    max_relative_position=14,
                                    relative_position=True,
                                    change_qkv=False,
                                    abs_pos=True)
    model.config = config
    model._initialize_alphas()
    # load the checkpoint
    ckpt = torch.load(
        "/work/dlclarge1/sukthank-transformer_search/code/OneShotNASwithWE/output_cifar10_darts_vgp_with_relu/checkpoint_99.pth",
        map_location='cpu')
    temp_dict = {}
    for k in ckpt["model"].keys():
        if k == "patch_embed_super.cls_token":
            temp_dict["cls_token"] = ckpt["model"][k]
        elif k == "patch_embed_super.pos_embed":
            temp_dict["pos_embed"] = ckpt["model"][k]
        else:
            temp_dict[k] = ckpt["model"][k]
    model.load_state_dict(temp_dict)
    # sample a random configuration
    config_sampled = sample_configs(config)
    # Set the config (ie subsample an architecture)
    model.set_sample_config(config_sampled)
    # Test forward pass
    input = torch.randn([2, 3, 224, 224])
    out = model(input)
    #print(out.shape)
    # Print the architecture parameters
    model.print_alphas()
