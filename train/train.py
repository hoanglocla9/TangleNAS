from collections import namedtuple
import time
import torch

from search_spaces.DARTS.utils import AverageMeter
from search_spaces.NB201.utils.checkpoints import copy_checkpoint, save_checkpoint
from search_spaces.NB201.utils.time import time_string
from search_spaces.NB201.utils import calc_accuracy
from train.utils import distill
import logging

TrainingMetrics = namedtuple("TrainingMetrics", "loss acc_top1 acc_top5")


class Trainer:

    def __init__(self,
                 model,
                 data,
                 model_optimizer,
                 scheduler,
                 criterion,
                 logger,
                 batch_size,
                 use_data_parallel=False,
                 print_freq=20,
                 drop_path_prob = 0.1,
                 load_saved_model:bool = False):
        self.model = model
        self.model_optimizer = model_optimizer
        self.scheduler = scheduler
        self.data = data
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.logger = logger
        self.criterion = criterion
        self.use_data_parallel = use_data_parallel
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.load_saved_model = load_saved_model
        self.drop_path_prob = drop_path_prob

    def _load_onto_data_parallel(self, network, criterion):
        if torch.cuda.is_available():
            network, criterion = torch.nn.DataParallel(
                self.model).cuda(), criterion.cuda()

        return network, criterion

    def _save_checkpoint(self, epoch, is_best):
        save_path = save_checkpoint(
            state={
                "epoch":
                epoch + 1,
                "model":
                self.model.state_dict(),
                "w_optimizer":
                self.model_optimizer.state_dict(),
                "w_scheduler":
                self.scheduler.state_dict()
                if self.scheduler is not None else [],
                "test_losses":
                self.test_losses,
                "test_accs_top1":
                self.test_accs_top1,
                "test_accs_top5":
                self.test_accs_top5,
                "train_losses":
                self.train_losses,
                "train_accs_top1":
                self.train_accs_top1,
                "train_accs_top5":
                self.train_accs_top5
            },
            filename=self.logger.path('model'),
            logger=self.logger,
        )

        _ = save_checkpoint(
            state={
                "epoch": epoch + 1,
                # "args": deepcopy(config), #TODO: find a way to save args
                "last_checkpoint": save_path,
                "test_losses": self.test_losses,
                "test_accs_top1": self.test_accs_top1,
                "test_accs_top5": self.test_accs_top5,
                "train_losses": self.train_losses,
                "train_accs_top1": self.train_accs_top1,
                "train_accs_top5": self.train_accs_top5,
            },
            filename=self.logger.path("info"),
            logger=self.logger,
        )

        if is_best == True:
            copy_checkpoint(self.logger.path('model'),
                            self.logger.path('best'), self.logger)

    def _init_empty_model_state_info(self):
        self.start_epoch = 0
        self.test_losses = {}
        self.train_losses = {}
        self.train_accs_top1 = {}
        self.train_accs_top5 = {}
        self.test_accs_top1 = {"best": -1}
        self.test_accs_top5 = {}

    def _load_model_state_if_exists(self):
        last_info = self.logger.path("info")

        if last_info.exists():  # automatically resume from previous checkpoint
            self.logger.log(
                "=> loading checkpoint of the last-info '{:}' start".format(
                    last_info))
            last_info = torch.load(last_info)
            self.start_epoch = last_info["epoch"]
            checkpoint = torch.load(last_info["last_checkpoint"])
            self.test_accs_top1 = checkpoint["test_accs_top1"]
            self.test_losses = checkpoint["test_losses"]
            self.test_accs_top5 = checkpoint["test_accs_top5"]
            self.train_losses = checkpoint["train_losses"]
            self.train_accs_top1 = checkpoint["train_accs_top1"]
            self.train_accs_top5 = checkpoint["train_accs_top5"]

            self.model.load_state_dict(checkpoint["model"])
            self.scheduler.load_state_dict(checkpoint["w_scheduler"])
            self.model_optimizer.load_state_dict(checkpoint["w_optimizer"])
            self.logger.log(
                "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch."
                .format(last_info, self.start_epoch))

            return True

        self.logger.log(
            "=> did not find the last-info file : {:}".format(last_info))
        return False

    def train(self, epochs):
        if self.use_data_parallel == True:
            network, criterion = self._load_onto_data_parallel(
                self.model, self.criterion)
        else:
            network, criterion = self.model.to(self.device), self.criterion.to(
                self.device)

        if self.load_saved_model:
            load_model = self._load_model_state_if_exists()
        else:
            load_model = False

        if not load_model:
            self._init_empty_model_state_info()

        start_time, train_time, epoch_time = time.time(), AverageMeter(
        ), AverageMeter(),
        train_loader, _, test_loader = self.data.get_dataloaders(
            batch_size=self.batch_size)
        self.logger.log(len(train_loader)*256)
        for epoch in range(self.start_epoch, epochs):
            epoch_str = f"{epoch:03d}-{epochs:03d}"
            #self.scheduler.update(epoch, 0.0)
            self.model.before_epoch(self.drop_path_prob * epoch / epochs)
            base_metrics = self.train_epoch(train_loader, network, criterion,
                                            self.scheduler,
                                            self.model_optimizer, epoch_str,
                                            self.print_freq, self.logger)

            train_time.update(time.time() - start_time)
            self.logger.log_metrics('Train: Model metrics', base_metrics,
                                    epoch_str, train_time.sum)

            test_metrics = self.valid_func(test_loader, self.model,
                                           self.criterion)
            
            self.logger.log_metrics('Evaluation:', test_metrics, epoch_str)

            self.test_losses[epoch], self.test_accs_top1[
                epoch], self.test_accs_top5[epoch] = test_metrics
            self.train_losses[epoch], self.train_accs_top1[
                epoch], self.train_accs_top5[epoch] = base_metrics

            if test_metrics.acc_top1 > self.test_accs_top1["best"]:
                self.test_accs_top1["best"] = test_metrics.acc_top1
                self.logger.log(
                    "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%."
                    .format(epoch_str, test_metrics.acc_top1))
                is_best = True
            else:
                is_best = False

            # save checkpoint
            self._save_checkpoint(epoch, is_best)

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

    def train_epoch(self, train_loader, network, criterion, w_scheduler,
                    w_optimizer, epoch_str, print_freq, logger):
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(
        ), AverageMeter()
        network.train()
        end = time.time()

        for step, (base_inputs, base_targets) in enumerate(train_loader):

            base_targets = base_targets.to(self.device)
            base_inputs = base_inputs.to(self.device)

            # measure data loading time
            data_time.update(time.time() - end)

            # update the model weights
            w_optimizer.zero_grad()
            _, logits_aux, logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            if logits_aux is not None:
               loss_aux = criterion(logits_aux, base_targets)
               base_loss += network.auxiliary_weight*loss_aux
            base_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.get_weights(),
                5)  # TODO: Does this vary with the one-shot optimizers?
            w_optimizer.step()

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
                #break
        base_metrics = TrainingMetrics(base_losses.avg, base_top1.avg,
                                       base_top5.avg)

        return base_metrics

    def valid_func(self, test_loader, network, criterion):
        test_losses, test_top1, test_top5 = AverageMeter(), AverageMeter(
        ), AverageMeter()
        network.eval()

        with torch.no_grad():
            for step, (test_inputs, test_targets) in enumerate(test_loader):
                if torch.cuda.is_available():
                    test_targets = test_targets.cuda(non_blocking=True)
                    test_inputs = test_inputs.cuda(non_blocking=True)

                # prediction
                _,_, logits = network(test_inputs)
                test_loss = criterion(logits, test_targets)

                # record
                test_prec1, test_prec5 = calc_accuracy(logits.data,
                                                       test_targets.data,
                                                       topk=(1, 5))

                test_losses.update(test_loss.item(), test_inputs.size(0))
                test_top1.update(test_prec1.item(), test_inputs.size(0))
                test_top5.update(test_prec5.item(), test_inputs.size(0))

        return TrainingMetrics(test_losses.avg, test_top1.avg, test_top5.avg)

    def _update_meters(self, inputs, logits, targets, loss, loss_meter,
                       top1_meter, top5_meter):
        base_prec1, base_prec5 = calc_accuracy(logits.data,
                                               targets.data,
                                               topk=(1, 5))
        loss_meter.update(loss.item(), inputs.size(0))
        top1_meter.update(base_prec1.item(), inputs.size(0))
        top5_meter.update(base_prec5.item(), inputs.size(0))
