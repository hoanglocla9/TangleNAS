import argparse
from enum import Enum
import random
import numpy as np

from datasets.data import (
    CIFAR100Data,
    CIFAR10Data,
    ExperimentData,
    ImageNet16120Data,
    FASHIONMNISTData
)
# TODO: Handle v1 and v2
from search_spaces.NATS.model_search_v1 import NATSSearchSpaceV1
from search_spaces.NATS.model_search_v2 import NATSSearchSpaceV2
from search_spaces.NATS.genotypes import Structure as CellStructure
from search_spaces.NB201.model_search import NASBench201SearchSpace
from search_spaces.NB201.utils.config import load_config
from search_spaces.NB201.utils.logging import Logger
from search_spaces.NB201.operations import NAS_BENCH_201 as NB201_SEARCH_SPACE
from search_spaces.NB201.utils.optimizers import CrossEntropyLabelSmooth
from search_spaces.DARTS.model_search import DARTSSearchSpace
from toy_search_spaces.cell_topology.model_search import ToyCellSearchSpace
from toy_search_spaces.conv_macro.model_search import ConvNetMacroSpace
from search.architects import ArchitectV1, ArchitectV2, DummyArchitect

import torch
import torch.backends.cudnn as cudnn
import wandb

from search.search import OneShotSearch


class SearchSpace(Enum):
    DARTS = 'darts'
    NB201 = 'nb201'
    NATS_V1 = 'nats_v1'
    NATS_V2 = 'nats_v2'
    TOY_CELL = 'toy_cell'
    TOY_CELL_STACKED = 'toy_cell_stacked'
    TOY_CONV_MACRO = 'toy_conv_macro'


class OneShotOptimizer(Enum):
    DARTS_V1 = 'darts_v1'
    DARTS_V2 = 'darts_v2'
    DRNAS = 'drnas'
    GDAS = 'gdas'
    SPOS = 'spos'


class DatasetType(Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    IMGNET = 'imgnet'
    IMGNET16_1K = 'imgnet16_1k'
    IMGNET16_120 = 'imgnet16_120'
    FASHION_MNIST = 'fashion_mnist'


N_CLASSES = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100,
    DatasetType.IMGNET: 1000,
    DatasetType.IMGNET16_1K: 1000,
    DatasetType.IMGNET16_120: 120,
    DatasetType.FASHION_MNIST: 10,
}


class Experiment:

    def __init__(self,
                 search_space: SearchSpace,
                 one_shot_opt: OneShotOptimizer,
                 dataset: DatasetType,
                 datapath: str,
                 seed: int,
                 log_dir: str,
                 name: str = None,
                 print_freq: int = 20,
                 load_saved_model: bool = False,
                 path_to_benchmark: str = '',
                 entangle_weights: bool = True,
                 use_we_v2: bool = False,
                 submission: bool = False,
                 save_every_epoch: bool = False):
        self.search_space = search_space
        self.one_shot_opt = one_shot_opt
        self.dataset = dataset
        self.datapath = datapath
        self.log_dir = log_dir
        self.seed = seed
        self.print_freq = print_freq
        self.load_saved_model = load_saved_model
        self.path_to_benchmark = path_to_benchmark
        self.entangle_weights = entangle_weights
        self.use_we_v2 = use_we_v2
        self.save_every_epoch = save_every_epoch

        if name is not None:
            self.name = f'{name}_{self.seed}'
        else:
            we_str = 'we_' if self.entangle_weights else 'ws_'
            we_str = 'we2_' if self.use_we_v2 else we_str
            self.name = f'{we_str}{self.search_space.value}_{self.one_shot_opt.value}_{self.dataset.value}_{self.seed}'

        # TODO: Do different datasets have different configs?
        self.config_file_name = f'{self.search_space.value}-{self.one_shot_opt.value}.config'
        self.config_path = f'./configs/search/{self.config_file_name}'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = N_CLASSES[self.dataset]
        self.submission = submission

    def _get_optimizer(self, parameters, config):
        assert (hasattr(config, "optim")), "optim key missing in config"
        if config.optim == "SGD":
            optim = torch.optim.SGD(
                parameters,
                config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
            )
        elif config.optim == "RMSprop":
            optim = torch.optim.RMSprop(parameters,
                                        config.lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        elif config.optim == "Adam":
            optim = torch.optim.Adam(parameters,
                                     betas=(0.9, 0.995), weight_decay=5e-4, lr=config.lr)
        else:
            raise ValueError(f"Invalid optim: {config.optim}")

        return optim

    def _get_criterion(self, config):
        if config.criterion == "Softmax":
            criterion = torch.nn.CrossEntropyLoss()
        elif config.criterion == "SmoothSoftmax":
            criterion = CrossEntropyLabelSmooth(config.class_num,
                                                config.label_smooth)
        else:
            raise ValueError("invalid criterion : {:}".format(
                config.criterion))

        return criterion

    def _get_data(self, cutout, train_portion) -> ExperimentData:
        if self.dataset == DatasetType.CIFAR10:
            return CIFAR10Data(self.datapath,
                               cutout=cutout,
                               train_portion=train_portion)
        elif self.dataset == DatasetType.CIFAR100:
            return CIFAR100Data(self.datapath,
                                cutout=cutout,
                                train_portion=train_portion)
        elif self.dataset == DatasetType.IMGNET16_120:
            return ImageNet16120Data(self.datapath,
                                     cutout=cutout,
                                     train_portion=train_portion)
        elif self.dataset == DatasetType.FASHION_MNIST:
            return FASHIONMNISTData(self.datapath, train_portion=train_portion)
        else:
            # TODO: Handle other cases
            raise ValueError(f'{self.dataset} not known')

    def _get_architect(self, model, config=None):
        grad_weights = None
        if self.one_shot_opt == OneShotOptimizer.DARTS_V2:
            return ArchitectV2(model=model, config=config, grad_weights=grad_weights)
        elif self.one_shot_opt == OneShotOptimizer.SPOS:
            return DummyArchitect(model=model, config=config, grad_weights=grad_weights)
        elif self.one_shot_opt in (OneShotOptimizer.DARTS_V1,
                                   OneShotOptimizer.GDAS):
            return ArchitectV1(model=model, config=config, grad_weights=grad_weights)
        elif self.one_shot_opt == OneShotOptimizer.DRNAS:
            use_kl_loss = config.reg_type == "kl"
            if use_kl_loss == True:
                reg_scale = 0
            else:
                reg_scale = config.reg_scale
            return ArchitectV1(model=model,
                               config=config,
                               use_kl_loss=use_kl_loss,
                               reg_scale=reg_scale, grad_weights=grad_weights)
        elif self.one_shot_opt == OneShotOptimizer.SPOS:
            return None
        else:
            raise ValueError(f'{self.one_shot_opt} not known')

    def get_search_model(self, config, criterion):

        criterion = criterion.to(self.device)
        reg_type = 'l2' if not hasattr(config, 'reg_type') else config.reg_type
        reg_scale = 1e-3 if not hasattr(config,
                                        'reg_scale') else config.reg_scale

        if self.search_space == SearchSpace.NB201:
            search_model = NASBench201SearchSpace(
                optimizer_type=self.one_shot_opt.value,
                C=config.channel,
                N=config.num_cells,
                max_nodes=config.max_nodes,
                num_classes=self.num_classes,
                search_space=NB201_SEARCH_SPACE,
                affine=config.affine,
                track_running_stats=config.track_running_stats,
                criterion=criterion,
                reg_type=reg_type,
                reg_scale=reg_scale,
                path_to_benchmark=self.path_to_benchmark,
                entangle_weights=self.entangle_weights,
                use_we_v2=self.use_we_v2)
        elif self.search_space == SearchSpace.DARTS:
            search_model = DARTSSearchSpace(
                optimizer_type=self.one_shot_opt.value,
                C=config.channel,
                num_classes=self.num_classes,
                layers=config.layers,
                criterion=criterion,
                reg_type=reg_type,
                reg_scale=reg_scale,
                entangle_weights=self.entangle_weights,
                use_we_v2=self.use_we_v2)
        elif self.search_space == SearchSpace.NATS_V1:  # TODO: Handle v1 and v2
            genotype = CellStructure.str2structure(
                '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
            )
            search_model = NATSSearchSpaceV1(
                optimizer_type=self.one_shot_opt.value,
                genotype=genotype,
                num_classes=self.num_classes,
                criterion=criterion,
                reg_type=reg_type,
                reg_scale=reg_scale,
                affine=config.affine,
                track_running_stats=config.track_running_stats,
                path_to_benchmark=self.path_to_benchmark
            )

        elif self.search_space == SearchSpace.NATS_V2:  # TODO: Handle v1 and v2
            genotype = CellStructure.str2structure(
                '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
            )
            search_model = NATSSearchSpaceV2(
                optimizer_type=self.one_shot_opt.value,
                genotype=genotype,
                num_classes=self.num_classes,
                criterion=criterion,
                reg_type=reg_type,
                reg_scale=reg_scale,
                affine=config.affine,
                track_running_stats=config.track_running_stats,
                path_to_benchmark=self.path_to_benchmark,
                use_we_v2=self.use_we_v2
            )
        elif self.search_space == SearchSpace.TOY_CELL:
            search_model = ToyCellSearchSpace(
                optimizer_type=self.one_shot_opt.value,
                criterion=criterion,
                num_classes=self.num_classes,
                max_nodes=config.max_nodes,
                api=self.path_to_benchmark,
                num_channels=config.channel,
                entangle_weights=self.entangle_weights,
                use_we_v2=self.use_we_v2,
            )
        elif self.search_space == SearchSpace.TOY_CONV_MACRO:
            search_model = ConvNetMacroSpace(
                optimizer_type=self.one_shot_opt.value,
                use_we_v2=self.use_we_v2,
                api=self.path_to_benchmark,
            )
            print(search_model.api)
        return search_model

    def get_config(self):
        config = load_config(path=self.config_path,
                             extra={},
                             logger=self.logger)
        return config

    def _update_sampler(self, sampler, one_shot_optimizer, config):
        if one_shot_optimizer == OneShotOptimizer.GDAS:
            sampler.set_taus(tau_min=config.tau_min, tau_max=config.tau_max)
            sampler.set_total_epochs(config.epochs)

    def set_seed(self, rand_seed):
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.manual_seed(rand_seed)

        cudnn.deterministic = True
        torch.manual_seed(rand_seed)
        cudnn.enabled = True

    def run(self, watch_model=False):
        project_name = "tanglenas-submission" if self.submission is True else "tanglenas-prototype"

        wandb.init(project=project_name,
                   entity="nas-team-freiburg", name=self.name)
        self.logger = Logger(log_dir=self.log_dir,
                             seed=self.seed,
                             exp_name=f'{self.name}-{wandb.run.id}')

        config = self.get_config()
        self.set_seed(self.seed)

        if self.search_space == SearchSpace.DARTS:
            sample_subset = True
        else:
            sample_subset = False
        criterion = self._get_criterion(config)
        model = self.get_search_model(config=config, criterion=criterion)

        # Weights and biases logging
        wandb.config.update(config._asdict())
        wandb.config.seed = self.seed

        if watch_model is True:
            wandb.watch(model)

        self._update_sampler(model.sampler, self.one_shot_opt, config)

        architect = self._get_architect(model, config)
        data = self._get_data(config.cutout, config.train_portion)

        w_optimizer = self._get_optimizer(model.get_weights(), config)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=w_optimizer,
            T_max=float(config.epochs),
            eta_min=config.learning_rate_min)

        searcher = OneShotSearch(search_model=model,
                                 data=data,
                                 architect=architect,
                                 model_optimizer=w_optimizer,
                                 scheduler=w_scheduler,
                                 criterion=criterion,
                                 logger=self.logger,
                                 batch_size=config.batch_size,
                                 use_data_parallel=config.use_data_parallel,
                                 print_freq=self.print_freq,
                                 load_saved_model=self.load_saved_model,
                                 sample_subset=sample_subset,
                                 save_every_epoch=self.save_every_epoch,)

        searcher.search(config.epochs)

        return searcher


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'One shot optimization with weight entanglement', add_help=False)
    parser.add_argument('--searchspace',
                        default='toy_cell',
                        help='search space in (darts, nb201, nats, toy_cell)',
                        type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument(
        '--optimizer',
        help='name of optimizer (darts_v1, darts_v2, gdas, drnas, spos)',
        default='darts_v1',
        type=str)
    parser.add_argument(
        '--logdir', default='./final_experiments_nats', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--load_saved_model', action='store_true', default=False,
                        help='Load the saved models before searching the supernet')
    parser.add_argument('--path_to_benchmark', type=str,
                        default='/work/dlclarge2/sukthank-tanglenas/merge/TangleNAS-dev/toy_search_spaces/conv_macro/results/conv_macro_bench.pkl', help='Path to benchmark')
    parser.add_argument('--data_path', type=str,
                        default='.', help='Path to dataset')
    parser.add_argument('--no_weight_entanglement', default=False,
                        action='store_true', help='Use this flag to disable weight entanglement')
    parser.add_argument('--use_we_v2', default=False, action='store_true',
                        help='Use this flag to use weight entanglement version 2. If used, this flag overrides no_weight_entanglement')
    parser.add_argument('--submission', default=False, action='store_true',
                        help='Use tangle-nas submission wandb project to track the experiment')
    parser.add_argument('--watch_model', default=False,
                        action='store_true', help='watch the model on wandb')
    parser.add_argument('--save_every_epoch', default=False,
                        action='store_true', help='checkpoint every epoch')

    args = parser.parse_args()

    searchspace = SearchSpace(args.searchspace)
    dataset = DatasetType(args.dataset)
    optimizer = OneShotOptimizer(args.optimizer)

    experiment = Experiment(searchspace,
                            one_shot_opt=optimizer,
                            dataset=dataset,
                            datapath=args.data_path,
                            seed=args.seed,
                            log_dir=args.logdir,
                            load_saved_model=args.load_saved_model,
                            path_to_benchmark=args.path_to_benchmark,
                            entangle_weights=not args.no_weight_entanglement,
                            use_we_v2=args.use_we_v2,
                            submission=args.submission,
                            save_every_epoch=args.save_every_epoch)
    searcher = experiment.run(args.watch_model)
