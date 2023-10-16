##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from pathlib import Path
import sys, time


class PrintLogger(object):

    def __init__(self):
        """Create a summary writer logging to log_dir."""
        self.name = "PrintLogger"

    def log(self, string):
        print(string)

    def close(self):
        print("-" * 30 + " close printer " + "-" * 30)


class Logger(object):

    def __init__(self,
                 log_dir,
                 seed,
                 create_model_dir=True,
                 use_tf=False,
                 exp_name=''):
        """Create a summary writer logging to log_dir."""
        self.seed = int(seed)
        self.log_dir = Path(log_dir) / exp_name
        self.model_dir = Path(log_dir) / ("checkpoint/" + exp_name)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        # self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        self.use_tf = bool(use_tf)
        self.tensorboard_dir = self.log_dir / ("tensorboard-{:}".format(
            time.strftime("%d-%h", time.gmtime(time.time()))))
        # self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime( '%d-%h-at-%H:%M:%S', time.gmtime(time.time()) )))
        self.logger_path = self.log_dir / "seed-{:}-T-{:}.log".format(
            self.seed,
            time.strftime("%d-%h-at-%H-%M-%S", time.gmtime(time.time())))
        self.logger_file = open(self.logger_path, "w")

        if self.use_tf:
            self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
            self.writer = tf.summary.FileWriter(str(self.tensorboard_dir))
        else:
            self.writer = None

    def __repr__(self):
        return "{name}(dir={log_dir}, use-tf={use_tf}, writer={writer})".format(
            name=self.__class__.__name__, **self.__dict__)

    def path(self, mode, epoch=None):
        valids = ("model", "best", "info", "log", None)
        if epoch is not None:
            return self.model_dir / "seed-{:}-{:}.pth".format(self.seed, epoch)
        if mode is None:
            return self.log_dir
        elif mode == "model":
            return self.model_dir / "seed-{:}-basic.pth".format(self.seed)
        elif mode == "best":
            return self.model_dir / "seed-{:}-best.pth".format(self.seed)
        elif mode == "info":
            return self.log_dir / "seed-{:}-last-info.pth".format(self.seed)
        elif mode == "log":
            return self.log_dir
        else:
            raise TypeError("Unknow mode = {:}, valid modes = {:}".format(
                mode, valids))

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write("{:}\n".format(string))
            self.logger_file.flush()

    def log_metrics(self, title, metrics, epoch_str='', totaltime=None):
        msg = "[{:}] {} : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
            epoch_str,
            title,
            metrics.loss,
            metrics.acc_top1,
            metrics.acc_top5,
        )

        if totaltime is not None:
            msg += f', time-cost={totaltime:.1f} s'

        self.log(msg)
