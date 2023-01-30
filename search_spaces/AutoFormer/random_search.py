import sys
import os
import random

print(sys.path)
sys.path.append(os.path.join(sys.path[0], '..'))

from lib.datasets import build_dataset
from supernet_engine_inherit import evaluate
from supernet_train import get_args_parser
from model_autoformer.supernet_transformer import Vision_TransformerSuper
import torch
import numpy as np
from torch.utils.data import Subset

config = {
    "embed_dim": [192, 216, 240],
    "mlp_ratio": [3.5, 4],
    "layer_num": [12, 13, 14],
    "num_heads": [3, 4]
}


class RandomSearchOptimizer(object):

    def __init__(self,
                 model,
                 config,
                 dataloader,
                 use_alphas=True,
                 device='cpu'):
        self.model = model
        self.config = config
        self.use_alphas = use_alphas
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.model.to(self.device)

    def search(self, save_path, n_iters=100):
        sampled_configs = []

        for it in range(n_iters):
            print(f'SAMPLING MODEL {it+1} of {n_iters}')

            try:
                sampled_config = self.model.sample(use_alphas=self.use_alphas)
                sampled_config['layer_num'] = int(sampled_config['layer_num'])

                self.model.set_sample_config(sampled_config)
                scores = self.evaluate(sampled_config, self.model)
                sampled_configs.append((sampled_config, scores))

                save_results(sampled_configs, save_path)

            except Exception as e:
                print('Caught exception', e)

        return sampled_configs

    def evaluate(self, config, model):
        result = evaluate(self.dataloader,
                          model,
                          self.device,
                          mode='not_super',
                          retrain_config=config)
        flops = model.get_complexity(256) / 10**9
        params = model.get_sampled_params_numel(config)

        result.update({'flops': float(flops), 'params': float(params)})

        return result


def load_model(n_classes, img_size, patch_size, model_path):

    model = Vision_TransformerSuper(img_size=img_size,
                                    patch_size=patch_size,
                                    embed_dim=240,
                                    depth=14,
                                    num_heads=3,
                                    mlp_ratio=4,
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
    ckpt = torch.load(model_path, map_location='cpu')
    temp_dict = {}
    for k in ckpt["model"].keys():
        if k == "patch_embed_super.cls_token":
            temp_dict["cls_token"] = ckpt["model"][k]
        elif k == "patch_embed_super.pos_embed":
            temp_dict["pos_embed"] = ckpt["model"][k]
        else:
            temp_dict[k] = ckpt["model"][k]
    model.load_state_dict(temp_dict)

    return model


def get_dataloader(args):
    dataset_train, args.nb_classes = build_dataset(is_train=False, args=args)

    dataset_train = Subset(dataset_train, np.arange(args.dataset_subset_size))
    sampler = torch.utils.data.RandomSampler(dataset_train)

    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    return dataloader


def test_model(model, args):
    # Test forward pass
    input = torch.randn([2, 3, args.input_size, args.input_size])
    out = model(input)
    print(out.shape)

    # Test to see if alphas are present
    model.print_alphas()

    # Test sampling a random architecture
    print('Randomly sampled model:')
    print(model.sample())


def save_results(results, path):
    with open(path, 'w') as f:
        f.write(str(results))


def get_save_dir(args):

    result_dir = os.path.join(
        '.', 'results', 'rs_oneshotwe', args.one_shot_optimizer, args.data_set,
        f'img_size_{args.input_size}', f'patch_size_{args.patch_size}',
        str(args.seed),
        'with_prior' if args.sample_with_priors else 'without_prior')

    return result_dir


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args(
    )  # Comment this line out to run locally without cmdline arguments

    # Uncomment the following lines to to run locally without passing in cmdline arguments
    # args = parser.parse_args(['--cfg', 'no_config_file'])
    # args.input_size = 32
    # args.patch_size = 2
    # args.device = 'cpu'
    # args.data_set = 'CIFAR100'
    # args.search_iters = 2
    # args.dataset_subset_size = 128

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    classes_label_counts = {'CIFAR10': 10, 'CIFAR100': 100, 'IMNET': 1000}

    nb_classes = classes_label_counts[args.data_set]

    model = load_model(nb_classes, args.input_size, args.patch_size,
                       args.model_path)
    sampled_config = model.sample()
    model.set_sample_config(sampled_config)

    test_model(model, args)

    dataloader = get_dataloader(args)
    optimizer = RandomSearchOptimizer(model,
                                      config,
                                      dataloader,
                                      use_alphas=args.sample_with_priors,
                                      device=args.device)

    result_dir = get_save_dir(args)
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, 'samples.json')
    results = optimizer.search(result_file, n_iters=args.search_iters)

    print('Search complete.')
