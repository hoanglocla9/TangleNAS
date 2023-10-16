import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
import time
from random import randint
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from lib.datasets import build_dataset
from supernet_engine import train_one_epoch, evaluate
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
from search_spaces.AutoFormer.model_one_shot.supernet_transformer_search_2 import Vision_TransformerSuper
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
from architect import Architect
import warnings
from timm_optimizer import create_optimizer
import wandb
warnings.filterwarnings('ignore')


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'layer_decay', None) is not None:
        kwargs['layer_decay'] = cfg.layer_decay
    if getattr(cfg, 'weight_decay', None) is not None:
        kwargs['weight_decay'] = cfg.weight_decay
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
def get_args_parser():
    parser = argparse.ArgumentParser(
        'AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # config file
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # custom parameters
    parser.add_argument('--platform',
                        default='pai',
                        type=str,
                        choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model',
                        default='',
                        type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position',
                        type=int,
                        default=14,
                        help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model',
                        default='',
                        type=str,
                        metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--mode',
                        type=str,
                        default='super',
                        choices=['super', 'retrain'],
                        help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop',
                        type=float,
                        default=0.0,
                        metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path',
                        type=float,
                        default=0.1,
                        metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block',
                        type=float,
                        default=None,
                        metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema',
                        action='store_false',
                        dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay',
                        type=float,
                        default=0.99996,
                        help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--rpe_type',
                        type=str,
                        default='bias',
                        choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt',
                        default='adamw',
                        type=str,
                        metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps',
                        default=1e-8,
                        type=float,
                        metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt-betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad',
                        type=float,
                        default=None,
                        metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched',
                        default='cosine',
                        type=str,
                        metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr',
                        type=float,
                        default=5e-4,
                        metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise',
                        type=float,
                        nargs='+',
                        default=None,
                        metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument(
        '--lr-noise-pct',
        type=float,
        default=0.67,
        metavar='PERCENT',
        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std',
                        type=float,
                        default=1.0,
                        metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr',
                        type=float,
                        default=1e-6,
                        metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min-lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power',
                        type=float,
                        default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs',
                        type=float,
                        default=30,
                        metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs',
                        type=int,
                        default=5,
                        metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--cooldown-epochs',
        type=int,
        default=10,
        metavar='N',
        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument(
        '--patience-epochs',
        type=int,
        default=10,
        metavar='N',
        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate',
                        '--dr',
                        type=float,
                        default=0.1,
                        metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter',
                        type=float,
                        default=0.4,
                        metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa',
                        type=str,
                        default='rand-m9-mstd0.5-inc1',
                        metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing',
                        type=float,
                        default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument(
        '--train-interpolation',
        type=str,
        default='bicubic',
        help=
        'Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false',
                        dest='repeated_aug')

    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob',
                        type=float,
                        default=0.25,
                        metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode',
                        type=str,
                        default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount',
                        type=int,
                        default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument(
        '--re:qlit',
        action='store_true',
        default=False,
        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.8,
        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument(
        '--cutmix',
        type=float,
        default=1.0,
        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument(
        '--cutmix-minmax',
        type=float,
        nargs='+',
        default=None,
        help=
        'cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup-prob',
        type=float,
        default=1.0,
        help=
        'Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup-switch-prob',
        type=float,
        default=0.5,
        help=
        'Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup-mode',
        type=str,
        default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )

    # Dataset parameters
    parser.add_argument(
        '--data-path',
        default=
        'imagenet/',
        type=str,
        help='dataset path')
    parser.add_argument('--data-set',
                        default='CIFAR10',
                        choices=['CIFAR100','CIFAR10','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str,
                        help='Image Net dataset path')
    parser.add_argument('--inat-category',
                        default='name',
                        choices=[
                            'kingdom', 'phylum', 'class', 'order',
                            'supercategory', 'family', 'genus', 'name'
                        ],
                        type=str,
                        help='semantic granularity')

    parser.add_argument('--output_dir',
                        default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--prior',
                        action='store_true',
                        help='Use optimizer with prior')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval',
                        action='store_true',
                        default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument(
        '--pin-mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no-pin-mem',
                        action='store_false',
                        dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.add_argument('--arch_learning_rate',
                        type=float,
                        default=3e-4,
                        metavar='LR',
                        help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay',
                        type=float,
                        default=1e-3,
                        metavar='LR',
                        help='weight decay for arch encoding')
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='local rank for distributed training')
    parser.add_argument('--unrolled',
                        action='store_true',
                        default=False,
                        help='use one-step unrolled validation loss')
    parser.add_argument('--use_we_v2',
                        action='store_true',
                        default=False,
                        help='Use we v2')
    parser.set_defaults(amp=True)

    #one shot optimizer args
    parser.add_argument('--tau_max', default=10, type=int, help='Tau max')
    parser.add_argument('--tau_min', default=0.1, type=int, help='Tau min')
    parser.add_argument('--one_shot_opt',
                        default="darts",
                        type=str,
                        help='Tau min')

    # Random Search parameters
    parser.add_argument('--search_iters',
                        default=100,
                        type=int,
                        help='Number of iterations of search to perform')

    parser.add_argument(
        '--dataset-subset-size',
        default=150,
        type=int,
        help=
        'Size of the dataset which should be used for evaluation in the search phase'
    )

    parser.add_argument(
        '--sample-with-priors',
        default=False,
        action='store_true',
        help=
        'If set to true, samples configs with probabilities as defined by the alpha weights'
    )

    parser.add_argument(
        '--model-path',
        default='./checkpoints/checkpoint_darts_cifar100.pth',
        type=str,
        help='Path to file with weights of trained supernetwork')

    parser.add_argument('--ratio',
                        default=0.5,
                        type=float,
                        help='train val split portion')

    return parser


def main(args):
    args.distributed = True
    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)
    #predictor = Net()
    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_test, _ = build_dataset(is_train=False, args=args)
    ratio = args.ratio
    len_train = len(dataset_train) - 10000
    len_val = 10000
    #data_path  = "/gpfs/bwfor/work/ws/fr_rs1131-tanglenas/subImageNet"
    #dataset_val, _ = build_dataset(is_train= True, args=args, data_path=data_path)
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train, [
            len_train,
            len_val
        ])
    print('train dataset length: ', len(dataset_train))
    print('val dataset length: ', len(dataset_val))
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train,
                                      num_replicas=num_tasks,
                                      rank=global_rank,
                                      shuffle=True)
            sampler_val = RASampler(dataset_val,
                                    num_replicas=num_tasks,
                                    rank=global_rank,
                                    shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                ConcatDataset(dataset_train,dataset_val),
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True)
            '''sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True)'''
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
    #dataset = torch.utils.data.TensorDataset(dataset_train, dataset_val)
    data_loader_train = torch.utils.data.DataLoader(
        ConcatDataset(dataset_train,dataset_val),
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=int(2 * args.batch_size),
        sampler=sampler_test,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=args.mixup,
                         cutmix_alpha=args.cutmix,
                         cutmix_minmax=args.cutmix_minmax,
                         prob=args.mixup_prob,
                         switch_prob=args.mixup_switch_prob,
                         mode=args.mixup_mode,
                         label_smoothing=args.smoothing,
                         num_classes=args.nb_classes)

    print(f"Creating SuperVisionTransformer")
    config = {
        "embed_dim": [192, 216, 240],
        "mlp_ratio": [3.5, 4 ],
        "layer_num": [12, 13, 14],
        "num_heads": [3,4]
    }
    model = Vision_TransformerSuper(
        optimizer=args.one_shot_opt,
        config=config,
        img_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=max(config["embed_dim"]),
        depth=max(config["layer_num"]),
        num_heads=max(config["num_heads"]),
        mlp_ratio=max(config["mlp_ratio"]),
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        use_we_v2=args.use_we_v2)
    #for n, p in model.named_parameters():
    #    print(n)
    columns = ['embed_dim', 'layer_num', 'epoch']
    for i in range(max(config["layer_num"])):
        columns.append("mlp_ratio_" + str(i))
        columns.append("num_heads_" + str(i))

    if args.teacher_model:
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.nb_classes,
        )
        teacher_model.to(device)
        teacher_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        teacher_model = None
        teacher_loss = None

    model_without_ddp = model
    if args.distributed:
        model.to(args.gpu)
        if args.one_shot_opt=="gdas":
            model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        model_without_ddp = model.module
    model.to("cuda")
    #predictor.to("cuda")
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size(
    ) / 512.0
    linear_scaled_arch_lr =  args.arch_learning_rate * args.batch_size * utils.get_world_size(
    ) / 512.0
    args.lr = linear_scaled_lr
    args.arch_learning_rate = linear_scaled_arch_lr
    optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()
    arch_loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    model_without_ddp._loss = criterion
    architect = Architect(model, model_without_ddp, args)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments
    with open(output_dir / "config.yaml", 'w') as f:
        f.write(args_text)
    arch_trajectory = {}
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            architect.optimizer.load_state_dict(checkpoint["arch_optimizer"])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    retrain_config = None
    if args.mode == 'retrain' and "RETRAIN" in cfg:
        retrain_config = {
            'layer_num': cfg.RETRAIN.DEPTH,
            'embed_dim': [cfg.RETRAIN.EMBED_DIM] * cfg.RETRAIN.DEPTH,
            'num_heads': cfg.RETRAIN.NUM_HEADS,
            'mlp_ratio': cfg.RETRAIN.MLP_RATIO
        }
    if args.eval:
        tau_curr = 0
        test_stats = evaluate(tau_curr,
                              data_loader_test,
                              model,
                              device,
                              amp=args.amp)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    print("Start training")
    import time
    start_time = time.time()
    max_accuracy = 0.0
    tau_curr = torch.Tensor([args.tau_max])
    tau_step = (args.tau_min - args.tau_max) / args.epochs
    model.module.sampler.set_taus(args.tau_min, args.tau_max)
    model.module.sampler.set_total_epochs(args.epochs)
    #for p in model.parameters():
    #    print(p.shape)
    for n,p in model.named_parameters():
        print(n)
    #if global_rank == 0:
    #    wandb_run_name = "autoformer_"+str(args.one_shot_opt)+"_"+str(args.data_set)+"_"+str(args.seed)+"_"+str(args.ratio)+"_"+str(start_time)
    #    wandb.init(project="autoformer", name=wandb_run_name)
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_val = None
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            #data_loader_val.sampler.set_epoch(epoch)
        model.module.sampler.before_epoch()
        #data_loader_val = None
        train_stats = train_one_epoch(tau_curr,
                                      args,
                                      architect,
                                      model,
                                      criterion,
                                      data_loader_train,
                                      data_loader_val,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      arch_loss_scaler,
                                      args.clip_grad,
                                      mixup_fn,
                                      amp=args.amp,
                                      teacher_model=teacher_model,
                                      teach_loss=teacher_loss)
        config = model.module.get_best_config()
        config["epoch"] = epoch
        print("best config", config)
        test_stats = evaluate(tau_curr,
                              data_loader_test,
                              model,
                              device,
                              amp=args.amp)
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        lr_scheduler.step(epoch)
        #architect.scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [
                str(output_dir) + "/" + 'checkpoint_' + str(epoch) + '.pth'
            ]
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'arch_optimizer': architect.optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        #if global_rank == 0:
        #    wandb.log({"train_loss": train_stats['loss']})
        #    wandb.log({"test_loss": test_stats['loss']})
        #    wandb.log({"test_acc": test_stats['acc1']})

        print(
            f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%"
        )
        
        print(f'Max accuracy: {max_accuracy:.2f}%')
        tau_curr += tau_step
        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if global_rank == 0:
            if max_accuracy == test_stats["acc1"]:
                arch_trajectory[str(epoch)] = config
                with open(args.output_dir +"/arch_trajectory.pkl", 'wb') as f:
                    pickle.dump(arch_trajectory, f)
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'AutoFormer training and evaluation script',
        parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
