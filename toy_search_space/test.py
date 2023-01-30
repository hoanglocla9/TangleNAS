from fvcore.common.config import CfgNode
import argparse
import sys
import os
import logging

logger = logging.getLogger(__name__)
from pathlib import Path


def pairwise(iterable):
    """
    Iterate pairwise over list.
    from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    """
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def load_config(path):
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config


def default_argument_parser():
    """
    Returns the argument parser with the default options.
    Inspired by the implementation of FAIR's detectron2
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, )
    parser.add_argument("--config-file",
                        default=None,
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="perform evaluation only")
    parser.add_argument("--seed", default=None, help="random seed")
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--model-path",
                        type=str,
                        default=None,
                        help="Path to saved model weights")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument("--rank",
                        default=0,
                        type=int,
                        help="node rank for distributed training")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:8888",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend",
                        default="nccl",
                        type=str,
                        help="distributed backend")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--datapath",
                        default=None,
                        metavar="FILE",
                        help="Path to the folder with train/test data folders")
    return parser


def parse_args(parser=default_argument_parser(), args=sys.argv[1:]):
    if "-f" in args:
        args = args[2:]
    return parser.parse_args(args)


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info("Experiment dir : {}".format(path))


def get_config_from_args(args=None, config_type="nas"):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.
    Prepares experiment directories.
    Args:
        args: args from a different argument parser than the default one.
    """

    if args is None:
        args = parse_args()
    logger.info("Command line args: {}".format(args))

    if args.config_file is None:
        config = load_default_config(config_type=config_type)
    else:
        config = load_config(path=args.config_file)

    # Override file args with ones from command line
    try:
        for arg, value in pairwise(args.opts):
            if "." in arg:
                arg1, arg2 = arg.split(".")
                config[arg1][arg2] = type(config[arg1][arg2])(value)
            else:
                config[arg] = type(
                    config[arg])(value) if arg in config else eval(value)

        config.eval_only = args.eval_only
        config.resume = args.resume
        config.model_path = args.model_path

        # load config file
        config.set_new_allowed(True)
        config.merge_from_list(args.opts)

    except AttributeError:
        for arg, value in pairwise(args):
            config[arg] = value

    # prepare the output directories
    if config_type == "nas":
        config.search.seed = config.seed
        config.evaluation.world_size = args.world_size
        config.gpu = config.search.gpu = config.evaluation.gpu = args.gpu
        config.evaluation.rank = args.rank
        config.evaluation.dist_url = args.dist_url
        config.evaluation.dist_backend = args.dist_backend
        config.evaluation.multiprocessing_distributed = args.multiprocessing_distributed
        config.save = "{}/{}/{}/{}/{}".format(config.out_dir,
                                              config.search_space,
                                              config.dataset, config.optimizer,
                                              config.seed)

    elif config_type == "bbo-bs":
        if not hasattr(config, 'evaluation'):
            config.evaluation = CfgNode()
        config.search.seed = config.seed
        config.evaluation.world_size = args.world_size
        config.gpu = config.search.gpu = config.evaluation.gpu = args.gpu
        config.evaluation.rank = args.rank
        config.evaluation.dist_url = args.dist_url
        config.evaluation.dist_backend = args.dist_backend
        config.evaluation.multiprocessing_distributed = args.multiprocessing_distributed

        if not hasattr(config, "config_id"):  #FIXME
            config.config_id = 0
        config.save = "{}/{}/{}/{}/config_{}/{}".format(
            config.out_dir, config.search_space, config.dataset,
            config.optimizer, config.config_id, config.seed)

    elif config_type == "predictor":
        config.search.seed = config.seed
        if config.predictor == "lcsvr" and config.experiment_type == "vary_train_size":
            config.save = "{}/{}/{}/{}_train/{}".format(
                config.out_dir,
                config.dataset,
                "predictors",
                config.predictor,
                config.seed,
            )
        elif config.predictor == "lcsvr" and config.experiment_type == "vary_fidelity":
            config.save = "{}/{}/{}/{}_fidelity/{}".format(
                config.out_dir,
                config.dataset,
                "predictors",
                config.predictor,
                config.seed,
            )
        else:
            config.save = "{}/{}/{}/{}/{}".format(
                config.out_dir,
                config.dataset,
                "predictors",
                config.predictor,
                config.seed,
            )
    elif config_type == "nas_predictor":
        config.search.seed = config.seed
        config.save = "{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.dataset,
            "nas_predictors",
            config.search_space,
            config.search.predictor_type,
            config.seed,
        )
    elif config_type == "oneshot":
        config.save = "{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.dataset,
            "nas_predictors",
            config.search_space,
            config.search.predictor_type,
            config.seed,
        )
    elif config_type == "statistics":
        config.save = "{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.search_space,
            config.dataset,
            "statistics",
            config.seed,
        )
    elif config_type == "zc":
        if not hasattr(config, 'search'):
            config.search = copy.deepcopy(config)
        if not hasattr(config, 'evaluation'):
            config.evaluation = CfgNode()

        if args.datapath is not None:
            config.train_data_file = os.path.join(args.datapath, 'train.json')
            config.test_data_file = os.path.join(args.datapath, 'test.json')
        else:
            config.train_data_file = None
            config.test_data_file = None

        config.save = "{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.config_type,
            config.search_space,
            config.dataset,
            config.predictor,
            config.seed,
        )

    else:
        print("invalid config type in utils/utils.py")

    config.data = "{}/data".format(get_project_root())

    create_exp_dir(config.save)
    create_exp_dir(config.save + "/search")  # required for the checkpoints
    create_exp_dir(config.save + "/eval")

    return config


args = parse_args(args=['--config-file', 'test.yaml'])
config = get_config_from_args(args)
print(config.seed)
conf = load_config("test.yaml")
print(conf.seed)
