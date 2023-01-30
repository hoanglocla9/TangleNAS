import pandas as pd
import pickle
with open(
        "/work/dlclarge2/sukthank-naslib_one_shot/OneShotNASwithWE/AutoFormerWE/output_cifar100_darts_wp_search/df_archs.pkl",
        "rb") as f:
    df = pickle.load(f)
last = df.iloc[-1]
import torch
from model_autoformer.supernet_transformer import Vision_TransformerSuper, derive_best_config

config = derive_best_config(last)
print(config)
model = Vision_TransformerSuper(img_size=32,
                                patch_size=2,
                                embed_dim=240,
                                depth=14,
                                num_heads=3,
                                mlp_ratio=4,
                                qkv_bias=True,
                                drop_rate=0,
                                drop_path_rate=0.1,
                                gp=True,
                                num_classes=10,
                                max_relative_position=14,
                                relative_position=True,
                                change_qkv=False,
                                abs_pos=True)
{
    'layer_num': 13,
    'mlp_ratio': [4, 3.5, 4, 4, 3.5, 4, 4, 4, 4, 4, 3.5, 3.5, 3.5],
    'num_heads': [3, 3, 4, 3, 3, 4, 3, 4, 3, 4, 4, 3, 4],
    'embed_dim':
    [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216]
}
model.set_sample_config(config)
print(model.get_complexity(256) / (10**9))
print(model.get_sampled_params_numel(config) / (10**6))
