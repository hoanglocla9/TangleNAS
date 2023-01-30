from .config import load_config, dict2config
from .logging import PrintLogger, Logger
from .optimizers import get_optimizer
from .datasets.datasets_with_transform import get_datasets
from .time import convert_secs2time, time_string
from .checkpoints import copy_checkpoint, prepare_logger, prepare_seed, save_checkpoint


def calc_accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
