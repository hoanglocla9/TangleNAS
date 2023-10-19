# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn as nn
import time
import torch
from tqdm import tqdm

from timm.utils import accuracy
from search_spaces.AutoFormer.lib.utils import MetricLogger
from search_spaces.MobileNetV3.utils import AverageMeter
from search_spaces.MobileNetV3.utils import (
    DistributedMetric,
    list_mean,
    val2list,
    MyRandomResizedCrop,
)
from search_spaces.MobileNetV3.imagenet_classification.manager import (
    DistributedRunManager
)

import wandb

__all__ = [
    "validate",
    "train_one_epoch",
    "train",
    "load_models"
]


def validate(
    run_manager,
    epoch=0,
    is_test=False,
    image_size=None,
    ks=None,
    expand_ratio=None,
    depth=None,
):
    dynamic_net = run_manager.net
    ddp_dynamic_net = run_manager.ddp_network

    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size is None:
        image_size = max(
            val2list(run_manager.run_config.data_provider.image_size, 1)
        )
    if ks is None:
        ks = max(dynamic_net.ks_list)
    if expand_ratio is None:
        expand_ratio = max(dynamic_net.expand_ratio_list)
    if depth is None:
        depth = max(dynamic_net.depth_list)

    snet_name = f"R{image_size}-D{depth}-E{expand_ratio}-K{ks}"

    supernet_settings = {
        "image_size": image_size,
        "d": depth,
        "e": expand_ratio,
        "ks": ks,
    }

    valid_log = ""

    run_manager.write_log(
        "-" * 30 + " Validate %s " % snet_name + "-" * 30, "train", should_print=False
    )
    run_manager.run_config.data_provider.assign_active_img_size(
        supernet_settings.pop("image_size")
    )
    dynamic_net.set_active_subnet(**supernet_settings)
    run_manager.write_log(dynamic_net.module_str, "train", should_print=False)

    run_manager.reset_running_statistics(dynamic_net)
    loss, (top1, top5) = run_manager.distributed_validate(
        epoch=epoch, is_test=is_test, run_str=snet_name, net=ddp_dynamic_net
    )

    valid_log += "%s (%.3f), " % (snet_name, top1)

    return (
        loss,
        top1,
        top5,
        valid_log,
    )

def optimize_model_weights(images, labels, run_manager):
    images, labels = images.cuda(), labels.cuda()

    # forward and backward pass
    output = run_manager.ddp_network(images)
    loss = run_manager.train_criterion(output, labels)
    loss.backward()

    # model optimizer step
    run_manager.optimizer.step()

    return output, loss

def optimize_simultaneous(images, labels, run_manager):
    # clear all the gradients
    run_manager.ddp_network.zero_grad()

    # optimize the model first
    output, loss = optimize_model_weights(images, labels, run_manager)

    # optimize the architecture weights
    run_manager.architect.optimizer.step()

    return output, loss

def optimizer_alternating(images, labels, images_search, labels_search, run_manager, args, epoch):
    # architect step on valid data
    images_search, labels_search = images_search.cuda(non_blocking=True), labels_search.cuda(non_blocking=True)

    run_manager.architect.step(
                            args,
                            epoch,
                            torch.nn.CrossEntropyLoss().cuda(),
                            images_search,
                            labels_search
                        )
    # clear ALL gradients
    run_manager.ddp_network.zero_grad()

    # model step on train data
    output, loss = optimize_model_weights(images, labels, run_manager)

    return output, loss


def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0, simultaneous_opt=False):

    dynamic_net = run_manager.ddp_network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()

    if distributed:
       run_manager.run_config.train_loader.sampler.set_epoch(epoch)

    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    metric_logger = MetricLogger(delimiter="  ")

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, (batch_1, batch_2) in enumerate(run_manager.run_config.train_loader):
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            
            # samples_search, targets_search = next(iter(run_manager.run_config.valid_loader))
            # samples_search, targets_search = samples_search.cuda(), targets_search.cuda()

            # # clean gradients
            # run_manager.architect.step(
            #                args,
            #                epoch,
            #                torch.nn.CrossEntropyLoss().cuda(),
            #                samples_search,
            #                targets_search)
            # #print(run_manager.net.arch_parameters())
            # images, labels = images.cuda(), labels.cuda()
            # #print(images.shape)
            # dynamic_net.zero_grad()
         
            # loss_of_subnets = []
            # # compute output
            # subnet_str = ""
            # subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, 0, 0))

            # output = dynamic_net(images)

            # loss = run_manager.train_criterion(output, labels)
            # loss_type = "ce"
            # loss_of_subnets.append(loss)
            # run_manager.update_metric(metric_dict, output, labels)
            # loss.backward()
            # run_manager.optimizer.step()

            # metric_logger.update(loss=loss.item())
            images , labels = batch_1
            images_search, labels_search  = batch_2
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            if simultaneous_opt is True:
                output, loss = optimize_simultaneous(images, labels, run_manager)
            else:
                output, loss = optimizer_alternating(images, labels, images_search, labels_search, run_manager, args, epoch)

            run_manager.update_metric(metric_dict, output, labels)
            metric_logger.update(loss=loss.item())

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            batch_size = images.shape[0]

            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            t.set_postfix(
                {
                    "loss": metric_logger.loss.avg,
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": "ce",
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
            #break
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return (
        metric_logger.loss.avg,
        [metric_logger.acc1.avg, metric_logger.acc5.avg]
    )

def save_model(run_manager, epoch, is_best=False, curr_acc=0, best_acc=0):
    run_manager.save_model(
        {
            "epoch": epoch,
            "curr_acc": curr_acc,
            "best_acc": best_acc,
            "optimizer": run_manager.optimizer.state_dict(),
            "arch_optimizer": run_manager.architect.optimizer.state_dict(),
            "state_dict": run_manager.network.state_dict(),
            "state_dict_archs": run_manager.network.state_dict_archs(),
            "genotype": run_manager.network.get_arch_represenation(),
        },
        is_best=is_best,
        model_name=run_manager.network.model_name,
        epoch=epoch
    )

def train(run_manager, args, validate_func=None):
    distributed = isinstance(run_manager, DistributedRunManager)

    if validate_func is None:
        validate_func = validate

    if args.one_shot_opt == "gdas":
        run_manager.network.sampler.set_taus(0.1,10)
        run_manager.network.sampler.set_total_epochs(run_manager.run_config.n_epochs + args.warmup_epochs)

    save_model(run_manager, -1, is_best=False, curr_acc=0, best_acc=0)

    for epoch in range(
        run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        if args.one_shot_opt == "gdas":
            run_manager.network.sampler.before_epoch()

        train_loss, (train_top1, train_top5) = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr, args.opt_strategy == "simultaneous"
        )

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_acc1, val_acc5, _val_log = validate_func(
                run_manager, epoch=epoch, is_test=False
            )

            run_manager.best_acc = max(run_manager.best_acc, val_acc1)

            if not distributed or run_manager.is_root:
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - args.warmup_epochs,
                        run_manager.run_config.n_epochs,
                        val_loss,
                        val_acc1,
                        run_manager.best_acc,
                    )
                )
                val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(
                    top1=train_top1, loss=train_loss
                )
                val_log += _val_log
                run_manager.write_log(val_log, "valid", should_print=False)

                state = {
                    "train_acc1": train_top1,
                    "train_acc5": train_top5,
                    "train_loss": train_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "val_loss": val_loss,
                    "epoch": epoch+1,
                    "genotype": str(run_manager.network.get_arch_represenation()),
                    "genotype": run_manager.network.get_arch_represenation(),
                }

                #wandb.log(state)

        save_model(
            run_manager,
            epoch,
            is_best=val_acc1==run_manager.best_acc,
            curr_acc=val_acc1,
            best_acc=run_manager.best_acc
        )


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    run_manager.write_log("Loaded init from %s" % model_path, "valid")