import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time


def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [
            random.choice(choices[dimension]) for _ in range(depth)
        ]

    config['embed_dim'] = [random.choice(choices['embed_dim'])] * depth

    config['layer_num'] = depth
    return config


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(tau_curr,
                    args,
                    architect,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader_train: Iterable,
                    data_loader_val: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    amp: bool = True,
                    teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if args.one_shot_opt == "spos":
        arch_params_sampled = model.module.sampler.sample_all_alphas(
            model.module._arch_parameters, tau_curr)
    else:
        arch_params_sampled = None
    for samples, targets in metric_logger.log_every(data_loader_train,
                                                    print_freq, header):
        samples_search, targets_search = next(iter(data_loader_val))
        samples_search = samples_search.to(device, non_blocking=True)
        targets_search = targets_search.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples_search, targets_search = mixup_fn(samples_search,
                                                      targets_search)
            samples, targets = mixup_fn(samples, targets)
        if args.one_shot_opt != "spos":
            architect.step(tau_curr,
                           args,
                           epoch,
                           amp,
                           criterion,
                           loss_scaler, [], [],
                           samples_search,
                           targets_search,
                           get_lr(optimizer),
                           optimizer,
                           unrolled=args.unrolled)
        del samples_search
        del targets_search
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples,
                                    tau_curr,
                                    arch_params_sampled=arch_params_sampled)
                    loss = 1 / 2 * criterion(
                        outputs, targets) + 1 / 2 * teach_loss(
                            outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples,
                                    tau_curr,
                                    arch_params_sampled=arch_params_sampled)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples,
                            tau_curr,
                            arch_params_sampled=arch_params_sampled)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs,
                                         targets) + 1 / 2 * teach_loss(
                                             outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss,
                        optimizer,
                        clip_grad=max_norm,
                        parameters=model.parameters(),
                        create_graph=is_second_order)
        else:
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
            optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(tau_curr, data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()
    if model.module.optimizer == "spos":
        arch_params_sampled = model.module.sampler.sample_all_alphas(
            model.module._arch_parameters, tau_curr)
    else:
        arch_params_sampled = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 1, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images, tau_curr)
                loss = criterion(output, target)
        else:
            output = model(images,
                           tau_curr,
                           arch_params_sampled=arch_params_sampled)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
