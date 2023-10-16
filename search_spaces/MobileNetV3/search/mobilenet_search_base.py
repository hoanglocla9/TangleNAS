# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import torch
import torch.backends.cudnn as cudnn
from search_spaces.MobileNetV3.search.architect import Architect
from search_spaces.MobileNetV3.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from search_spaces.MobileNetV3.model_search_ours import OFAMobileNetV3Mixture
from search_spaces.MobileNetV3.imagenet_classification.manager import DistributedImageNetRunConfig

from search_spaces.MobileNetV3.imagenet_classification.manager.distributed_run_manager import (
    DistributedRunManager,
)
from search_spaces.MobileNetV3.utils import MyRandomResizedCrop

import search_spaces.MobileNetV3.search.utils_train as utils

from search_spaces.MobileNetV3.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    train,
    validate
)

import wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="depth",
    choices=[
        "kernel",
        "depth",
        "expand",
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")
parser.add_argument("--one_shot_opt", type=str, default="drnas")
parser.add_argument("--opt_strategy", type=str, default="simultaneous")
parser.add_argument("--valid_size", type=float, default=0.8)

args = parser.parse_args()
if args.task == "kernel":
    args.path = "exp/normal2kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "depth":
    args.path = "exp/kernel2kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "2,3,4"
#elif args.task == "expand":
#    args.path = "exp/kernel_depth2kernel_depth_width/phase%d" % args.phase
#    args.dynamic_batch_size = 4
#if args.phase == 1:
#        args.n_epochs = 25
#        args.base_lr = 2.5e-3
##        args.warmup_epochs = 0
#        args.warmup_lr = -1
#        args.ks_list = "3,5,7"
#        args.expand_list = "4,6"
#        args.depth_list = "2,3,4"
#else:
args.n_epochs = 300
args.base_lr = 7.5e-3
args.warmup_epochs = 5
args.warmup_lr = -1
args.ks_list = "3,5,7"
args.expand_list = "3,4,6"
args.depth_list = "2,3,4"
args.manual_seed = 0
args.path = "experiments/mobilenet_drnas" #test"
args.seed = 9001
args.lr_schedule_type = "cosine"
args.world_size = 1
args.base_batch_size = 64

if args.opt_strategy == "simultaneous":
    args.valid_size = None

args.arch_learning_rate = 3e-4
args.arch_weight_decay = 1e-3 
args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10
args.dist_url = "env://"
args.n_worker = 0
args.resize_scale = 0.08
args.distort_color = "tf"
args.image_size = "128,160,192,224"
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = "proxyless"

args.width_mult_list = "1.0"
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = "ce"
args.device = "cuda"

args.save_path = "drnas_mbnet_test/"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

def main(args):
    os.makedirs(args.path, exist_ok=True)
    utils.init_distributed_mode(args)

    seed = args.seed + utils.get_rank()
    set_seed(seed)

    num_gpus = utils.get_world_size()
    global_rank = utils.get_rank()
    cudnn.benchmark = True

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate

    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedImageNetRunConfig(
        **args.__dict__, num_replicas=num_gpus
    )

    # print run config information
    if global_rank == 0:
        print("Run config:")
        for k, v in run_config.config.items():
            print("\t%s: %s" % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None

    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [
        float(width_mult) for width_mult in args.width_mult_list.split(",")
    ]
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )

    net = OFAMobileNetV3Mixture(
        n_classes=run_config.data_provider.n_classes,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        base_stage_width=args.base_stage_width,
        width_mult=args.width_mult_list,
        ks_list=args.ks_list,
        expand_ratio_list=args.expand_list,
        depth_list=args.depth_list,
        optimizer = args.one_shot_opt,
    ).cuda()

    net = torch.nn.parallel.DistributedDataParallel(
        module=net,
        find_unused_parameters=True if args.one_shot_opt == "gdas" else False
    )


    if global_rank == 0:
        project_name = "tanglenas-submission"
        name = f"we2_mobilenetv3_{args.one_shot_opt}_imagenet_{args.seed}"
        wandb.init(project=project_name, entity="nas-team-freiburg", name=name)
        wandb.config.update(args.__dict__)
        wandb.config.seed = args.seed
        wandb.watch(net.module)

    try:
        run_name = wandb.run.name
        run_id = wandb.run.id
    except Exception:
        run_name = f"mobilenetv3_{args.one_shot_opt}_{args.opt_strategy}_{args.seed}"
        run_id = ""

    model_name = f"{run_name}_{run_id}"
    net.module.model_name = model_name

    architect = Architect(net, args)

    distributed_run_manager = DistributedRunManager(
        args.path,
        net,
        architect,
        run_config,
        is_root=(global_rank == 0),
        args=args,
    )

    distributed_run_manager.save_config()

    validate_func_dict = {
        "image_size": args.image_size
            if isinstance(args.image_size, int)
            else max(args.image_size),
        "ks": max(args.ks_list),
        "expand_ratio": max(args.expand_list),
        "depth": max(args.depth_list),
    }

    def val_fn(_run_manager, epoch, is_test):
        return validate(_run_manager, epoch, is_test, **validate_func_dict)

    train(
        run_manager=distributed_run_manager,
        args=args,
        validate_func=val_fn
    )

if __name__ == "__main__":
    main(args)