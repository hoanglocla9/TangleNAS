from collections import namedtuple
import time
import torch
import wandb

from search_spaces.DARTS.utils import AverageMeter
from search_spaces.NB201.utils.checkpoints import copy_checkpoint, save_checkpoint
from search_spaces.NB201.utils.time import time_string
from search_spaces.NB201.utils import calc_accuracy
from search.architects import BaseArchitect
from train.utils import distill
import logging

TrainingMetrics = namedtuple("TrainingMetrics", "loss acc_top1 acc_top5")


class OneShotSearch:

    def __init__(self,
                 search_model,
                 data,
                 architect: BaseArchitect,
                 model_optimizer,
                 scheduler,
                 criterion,
                 logger,
                 batch_size,
                 use_data_parallel=False,
                 print_freq=20,
                 load_saved_model=False,
                 sample_subset=False):
        self.search_model = search_model
        self.model_optimizer = model_optimizer
        self.scheduler = scheduler
        self.data = data
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.architect = architect
        self.logger = logger
        self.criterion = criterion
        self.use_data_parallel = use_data_parallel
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.load_saved_model = load_saved_model
        self.sample_subset = sample_subset
        self.queried_results = []

    def _load_onto_data_parallel(self, network, criterion):
        if torch.cuda.is_available():
            network, criterion = torch.nn.DataParallel(self.search_model).cuda(), criterion.cuda()

        return network, criterion

    def _log_wandb(self, epoch):

        grad_norms = self.search_model.get_saved_stats()
        if 'grad_norms_flat' in grad_norms:
            grad_norms_flat = grad_norms['grad_norms_flat']

            for k, v in grad_norms_flat.items():
                grad_norms_flat[k] = v[-1]
        else:
            grad_norms_flat = {}

        state={
            "epoch": epoch + 1,
            "genotypes": [str(g) for g in self.genotypes.values()],
            "valid_losses": self.valid_losses[epoch],
            "valid_accuracies": self.valid_accs_top1[epoch],
            "valid_accs_top5": self.valid_accs_top5[epoch],
            "search_losses": self.search_losses[epoch],
            "search_accs_top1": self.search_accs_top1[epoch],
            "search_accs_top5": self.search_accs_top5[epoch],
            "gradient_stats": grad_norms_flat
        }
        if len(self.queried_results) > 0:
            state.update(self.queried_results[epoch])

        wandb.log(state)

    def _save_checkpoint(self, epoch, is_best):
        self._log_wandb(epoch)

        state={
            "epoch": epoch + 1,
            "genotypes": [str(g) for g in self.genotypes.values()],
            "valid_losses": self.valid_losses,
            "valid_accuracies": self.valid_accs_top1,
            "valid_accs_top5": self.valid_accs_top5,
            "search_losses": self.search_losses,
            "search_accs_top1": self.search_accs_top1,
            "search_accs_top5": self.search_accs_top5,
            "gradient_stats": self.search_model.get_saved_stats(),
            'queried_results': self.queried_results,
            "search_model": self.search_model.state_dict(),
            "w_optimizer": self.model_optimizer.state_dict(),
            "a_optimizer": self.architect.optimizer.state_dict(),
            "w_scheduler": self.scheduler.state_dict() if self.scheduler is not None else [],
        }

        save_path = save_checkpoint(
            state=state,
            filename=self.logger.path('model'),
            logger=self.logger,
        )

        if is_best == True:
            copy_checkpoint(self.logger.path('model'), self.logger.path('best'), self.logger)

    def _init_empty_model_state_info(self):
        self.genotypes = {-1: self.search_model.genotype()}
        self.start_epoch = 0
        self.valid_losses = {}
        self.search_losses = {}
        self.search_accs_top1 = {}
        self.search_accs_top5 = {}
        self.valid_accs_top1 = {"best": -1}
        self.valid_accs_top5 = {}

    def _load_model_state_if_exists(self):
        last_info = self.logger.path("info")

        if last_info.exists():  # automatically resume from previous checkpoint
            self.logger.log(
                "=> loading checkpoint of the last-info '{:}' start".format(
                    last_info))
            last_info = torch.load(last_info)
            self.start_epoch = last_info["epoch"]
            checkpoint = torch.load(last_info["last_checkpoint"])
            self.genotypes = checkpoint["genotypes"]
            self.valid_accs_top1 = checkpoint["valid_accuracies"]
            self.valid_losses = checkpoint["valid_losses"]
            self.valid_accs_top5 = checkpoint["valid_accs_top5"]
            self.search_losses = checkpoint["search_losses"]
            self.search_accs_top1 = checkpoint["search_accs_top1"]
            self.search_accs_top5 = checkpoint["search_accs_top5"]

            self.search_model.load_state_dict(checkpoint["search_model"])
            self.scheduler.load_state_dict(checkpoint["w_scheduler"])
            self.model_optimizer.load_state_dict(checkpoint["w_optimizer"])
            self.architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
            self.logger.log(
                "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch."
                .format(last_info, self.start_epoch)
            )

            return True

        self.logger.log(
            "=> did not find the last-info file : {:}".format(last_info))
        return False

    def search(self, epochs):
        if self.use_data_parallel == True:
            network, criterion = self._load_onto_data_parallel(self.search_model, self.criterion)
        else:
            network, criterion = self.search_model.to(self.device), self.criterion.to(self.device)

        if self.load_saved_model:
            load_model = self._load_model_state_if_exists()
        else:
            load_model = False

        if not load_model:
            self._init_empty_model_state_info()

        start_time, search_time, epoch_time = time.time(), AverageMeter(), AverageMeter()
        train_loader, val_loader, _ = self.data.get_dataloaders(batch_size=self.batch_size)

        for epoch in range(self.start_epoch, epochs):
            epoch_str = f"{epoch:03d}-{epochs:03d}"

            network.sampler.before_epoch()
            alphas = network.sampler.sample_epoch(network.arch_parameters(), sample_subset=self.sample_subset)
            base_metrics, arch_metrics = self.search_func(
                train_loader,
                val_loader,
                network,
                criterion,
                self.scheduler,
                self.model_optimizer,
                self.architect,
                alphas,
                epoch_str,
                self.print_freq,
                self.logger
            )

            search_time.update(time.time() - start_time)
            self.logger.log_metrics('Search: Model metrics', base_metrics, epoch_str, search_time.sum)
            self.logger.log_metrics('Search: Architecture metrics', arch_metrics, epoch_str)

            valid_metrics = self.valid_func(val_loader, self.search_model, self.criterion, alphas)
            self.logger.log_metrics('Evaluation:', valid_metrics, epoch_str)

            self.valid_losses[epoch], self.valid_accs_top1[epoch], self.valid_accs_top5[epoch] = valid_metrics
            self.search_losses[epoch], self.search_accs_top1[epoch], self.search_accs_top5[epoch] = base_metrics
            self.genotypes[epoch] = self.search_model.genotype()

            self.logger.log("<<<--->>> The {:}-th epoch : {:}".format(epoch_str, self.genotypes[epoch]))

            if self.search_model.api is not None:
                result = self.search_model.query()
                logging.info('{:}'.format(result))
                cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
                self.logger.log("cifar10 train {:.2f}% test {:.2f}%".format(cifar10_train, cifar10_test))
                self.logger.log(
                    "cifar100 train {:.2f}%  valid {:.2f}%  test {:.2f}%".
                    format(cifar100_train, cifar100_valid, cifar100_test)
                )
                self.logger.log(
                    "imagenet16 train  {:.2f}%  valid  {:.2f}%  test  {:.2f}% "
                    .format(imagenet16_train, imagenet16_valid,imagenet16_test)
                )

                results = {}
                results['benchmark_cifar10_train'] = cifar10_train
                results['benchmark_cifar10_test'] = cifar10_test
                results['benchmark_cifar100_train'] = cifar100_train
                results['benchmark_cifar100_valid'] = cifar100_valid
                results['benchmark_cifar100_test'] = cifar100_test
                results['benchmark_imagenet16_train'] = imagenet16_train
                results['benchmark_imagenet16_valid'] = imagenet16_valid
                results['benchmark_imagenet16_test'] = imagenet16_test
                self.queried_results.append(results)

            if valid_metrics.acc_top1 > self.valid_accs_top1["best"]:
                self.valid_accs_top1["best"] = valid_metrics.acc_top1
                self.genotypes["best"] = self.search_model.genotype()
                self.logger.log(
                    "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%."
                    .format(epoch_str, valid_metrics.acc_top1))
                is_best = True
            else:
                is_best = False

            # save checkpoint
            self._save_checkpoint(epoch, is_best)

            with torch.no_grad():
                self.logger.log("{:}".format(self.search_model.show_alphas()))

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

    def search_func(self, train_loader, valid_loader, network, criterion,
                    w_scheduler, w_optimizer, architect, alphas, epoch_str,
                    print_freq, logger):
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
        arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
        network.train()
        end = time.time()

        for step, (base_inputs, base_targets) in enumerate(train_loader):
            # scheduler.update(None, 1.0 * step / len(xloader)) # TODO: What was the point of this? and is it safe to remove?
            arch_inputs, arch_targets = next(iter(valid_loader))

            base_targets, arch_targets = base_targets.to(self.device), arch_targets.to(self.device)
            arch_inputs, base_inputs = arch_inputs.to(self.device), base_inputs.to(self.device)

            # measure data loading time
            data_time.update(time.time() - end)

            # update the architecture weights
            network.is_architect_step = True
            arch_loss, logits = architect.step(
                input_train=base_inputs,
                target_train=base_targets,
                input_valid=arch_inputs,
                target_valid=arch_targets,
                eta=w_scheduler.get_lr()[0],
                network_optimizer=w_optimizer,
            )
            network.is_architect_step = False

            self._update_meters(inputs=arch_inputs,
                                logits=logits,
                                targets=arch_targets,
                                loss=arch_loss,
                                loss_meter=arch_losses,
                                top1_meter=arch_top1,
                                top5_meter=arch_top5)

            # update the model weights
            w_optimizer.zero_grad()
            architect.optimizer.zero_grad()
            _, logits = network(base_inputs, alphas)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.get_weights(), 5)  # TODO: Does this vary with the one-shot optimizers?
            w_optimizer.step()
            w_optimizer.zero_grad()
            architect.optimizer.zero_grad()

            self._update_meters(inputs=base_inputs,
                                logits=logits,
                                targets=base_targets,
                                loss=base_loss,
                                loss_meter=base_losses,
                                top1_meter=base_top1,
                                top5_meter=base_top5)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % print_freq == 0 or step + 1 == len(train_loader):
                Sstr = ("*SEARCH* " + time_string() +
                        " [{:}][{:03d}/{:03d}]".format(epoch_str, step,
                                                       len(train_loader)))
                Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                    batch_time=batch_time, data_time=data_time)
                Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                    loss=base_losses, top1=base_top1, top5=base_top5)
                Astr = "Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                    loss=arch_losses, top1=arch_top1, top5=arch_top5)
                logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)
                #break
        base_metrics = TrainingMetrics(base_losses.avg, base_top1.avg, base_top5.avg)
        arch_metrics = TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

        return base_metrics, arch_metrics

    def valid_func(self, valid_loader, network, criterion, alphas):
        arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
        network.eval()

        with torch.no_grad():
            for step, (arch_inputs, arch_targets) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    arch_targets = arch_targets.cuda(non_blocking=True)
                    arch_inputs = arch_inputs.cuda(non_blocking=True)

                # prediction
                _, logits = network(arch_inputs, alphas)
                arch_loss = criterion(logits, arch_targets)

                # record
                arch_prec1, arch_prec5 = calc_accuracy(logits.data, arch_targets.data, topk=(1, 5))

                arch_losses.update(arch_loss.item(), arch_inputs.size(0))
                arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
                arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        return TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

    def _update_meters(self, inputs, logits, targets, loss, loss_meter,
                       top1_meter, top5_meter):
        base_prec1, base_prec5 = calc_accuracy(logits.data, targets.data, topk=(1, 5))
        loss_meter.update(loss.item(), inputs.size(0))
        top1_meter.update(base_prec1.item(), inputs.size(0))
        top5_meter.update(base_prec5.item(), inputs.size(0))
