import argparse
from enum import Enum
import random
import numpy as np

from datasets.data import CIFAR100Data, CIFAR10Data, ExperimentData, ImageNet16120Data
from search_spaces.NATS.model import NATSModel
from search_spaces.NATS.genotypes import Structure as CellStructure
from search_spaces.NB201.model import NASBench201Model
from search_spaces.NB201.utils.config import load_config
from search_spaces.NB201.utils.logging import Logger
from search_spaces.NB201.operations import NAS_BENCH_201 as NB201_SEARCH_SPACE
from search_spaces.NB201.utils.optimizers import CrossEntropyLabelSmooth
from search_spaces.DARTS.model import NetworkCIFAR
from search_spaces.DARTS.model import NetworkCIFAR
import search_spaces.DARTS.genotypes as genotypes
import torch
import torch.backends.cudnn as cudnn
from train.train  import Trainer


class SearchSpace(Enum):
    DARTS = 'darts'
    NB201 = 'nb201'
    NATS_V1 = 'nats_v1'
    NATS_V2 = 'nats_v2'


class DatasetType(Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    IMGNET = 'imgnet'
    IMGNET16_1K = 'imgnet16_1k'
    IMGNET16_120 = 'imgnet16_120'


N_CLASSES = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100,
    DatasetType.IMGNET: 1000,
    DatasetType.IMGNET16_1K: 1000,
    DatasetType.IMGNET16_120: 120
}


class Experiment:

    def __init__(self,
                 search_space: SearchSpace,
                 dataset: DatasetType,
                 datapath: str,
                 seed: int,
                 log_dir: str,
                 name: str = None,
                 print_freq: int = 20,
                 model_path: str = None,
                 one_shot_opt: str = None,
                 finetune: bool = False,
                 genotype: str = "darts",
                 load_saved_model: bool = False):
        self.search_space = search_space
        self.genotype = genotype
        self.dataset = dataset
        self.datapath = datapath
        self.log_dir = log_dir
        self.seed = seed
        self.print_freq = print_freq
        self.model_path = model_path
        self.finetune = finetune
        self.load_saved_model = load_saved_model
        if name is not None:
            self.name = f'{name}_{self.seed}'
        else:
            self.name = f'{self.search_space.value}_{one_shot_opt}_{self.dataset.value}_{self.seed}'

        self.logger = Logger(log_dir=self.log_dir,
                             seed=self.seed,
                             exp_name=self.name)
        # TODO: Do different datasets have different configs?
        self.config_file_name = f'{self.search_space.value}.config'
        if self.finetune:
            self.config_path = f'./configs/finetune/{self.config_file_name}'
        else:
            self.config_path = f'./configs/train/{self.config_file_name}'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = N_CLASSES[self.dataset]

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
        else:
            # TODO: Handle other cases
            raise ValueError(f'{self.dataset} not known')

    def get_search_model(self, config, criterion, model_path):

        criterion = criterion.to(self.device)

        if self.search_space == SearchSpace.NB201:
            model = NASBench201Model(
                C=config.channel,
                N=config.num_cells,
                max_nodes=config.max_nodes,
                num_classes=self.num_classes,
                search_space=NB201_SEARCH_SPACE,
                affine=config.affine,
                track_running_stats=config.track_running_stats)
            self.drop_path_prob = 0
        elif self.search_space == SearchSpace.DARTS:
            #C, num_classes, layers, auxiliary, genotype,
            # For darts no alg performs search on full fidelity hence train from scratch alsways
            model = NetworkCIFAR(config.channel, num_classes=self.num_classes, layers=config.layers, auxiliary=config.auxillary,  genotype=eval("genotypes.%s" % self.genotype), auxiliary_weight=config.auxiliary_weight)
            self.drop_path_prob = config.drop_path_prob
        elif self.search_space in [SearchSpace.NATS_V1, SearchSpace.NATS_V2]:
            genotype = CellStructure.str2structure(
                '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
            )
            model = NATSModel(genotype=genotype, num_classes=self.num_classes, affine=config.affine, track_running_stats=config.track_running_stats)
            self.drop_path_prob = 0
        else:
            raise Exception(f'Unknown search space {self.search_space}')
        last_info = torch.load(model_path)
        if self.finetune:
            
            print(last_info.keys())
            #checkpoint = torch.load(last_info["last_checkpoint"])
            model.load_state_dict(last_info["search_model"],strict=False)
        else:
            ckpt = last_info["search_model"]
            new_dict = {}
            for k in ckpt.keys():
                if "arch" in k or "alpha" in k:
                    new_dict[k] = ckpt[k]
            print(new_dict)
            model.load_state_dict(new_dict,strict=False)
        return model

    def get_config(self):
        print(self.config_path)
        config = load_config(path=self.config_path,
                             extra={},
                             logger=self.logger)
        return config

    def set_seed(self, rand_seed):
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.cuda.set_device(0)
        cudnn.benchmark = True
        torch.manual_seed(rand_seed)
        cudnn.enabled=True
        torch.cuda.manual_seed(rand_seed)

    def run(self):
        config = self.get_config()
        self.set_seed(self.seed)

        criterion = self._get_criterion(config)
        model = self.get_search_model(config=config,
                                      criterion=criterion,
                                      model_path=self.model_path)
        data = self._get_data(config.cutout, config.train_portion)

        w_optimizer = self._get_optimizer(model.get_weights(), config)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=w_optimizer,
            T_max=float(config.epochs),
            eta_min=config.learning_rate_min)

        trainer = Trainer(model=model,
                          data=data,
                          model_optimizer=w_optimizer,
                          scheduler=w_scheduler,
                          criterion=criterion,
                          logger=self.logger,
                          batch_size=config.batch_size,
                          use_data_parallel=config.use_data_parallel,
                          print_freq=self.print_freq,
                          drop_path_prob=self.drop_path_prob,
                          load_saved_model=self.load_saved_model)

        trainer.train(config.epochs)

        return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Fine tuning and training searched architectures', add_help=False)
    parser.add_argument('--searchspace',
                        default='nb201',
                        help='search space in (darts, nb201, nats)',
                        type=str)
    parser.add_argument('--one_shot_opt',
                        default='drnas',
                        help='optimizer used for search',
                        type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--logdir', default='./finetune_2', type=str)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--seed', default=444, type=int)
    parser.add_argument('--genotype', default="darts", type=str)
    parser.add_argument(
        '--model_path',
        default=
        ".",
        type=str)
    parser.add_argument('--load_saved_model', action='store_true', default=False, help='Load the saved models before training them')
    args = parser.parse_args()

    searchspace = SearchSpace(args.searchspace)
    dataset = DatasetType(args.dataset)

    experiment = Experiment(searchspace,
                            dataset=dataset,
                            datapath='.',
                            seed=args.seed,
                            log_dir=args.logdir,
                            model_path=args.model_path,
                            one_shot_opt=args.one_shot_opt,
                            finetune=args.finetune,
                            genotype=args.genotype,
                            load_saved_model=args.load_saved_model)
    trainer = experiment.run()
