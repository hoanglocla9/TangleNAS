import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import pickle
import utils as utils  # noqa: E402
from model import Network  # noqa: E402



def train(train_loader, model, criterion,
          optimizer, device):
    """
    Training loop. This function computes the DARTS loop, i.e. it takes one
    step in the architecture space and one in the weight space in an
    interleaving manner. For the architectural updates we use the validation
    set and for the search model parameters updates the training set. In DARTS
    these two sets have equal sizes, which in the case of MNIST it is 30k
    examples per each.
    """
    objs = utils.AvgrageMeter()
    accr = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        _accr = utils.accuracy(logits, target)
        objs.update(loss.item(), input.size(0))
        accr.update(_accr.item(), input.size(0))
        logging.info('train mini-batch %03d, loss=%e accuracy=%f', step,
                         objs.avg, accr.avg)
    return accr.avg, objs.avg


def infer(valid_loader, model, criterion, device):
    """
    Compute the accuracy on the validation set (the same used for updating the
    architecture).
    """
    objs = utils.AvgrageMeter()
    accr = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)

            logits = model(input)
            loss = criterion(logits, target)

            _accr = utils.accuracy(logits, target)
            objs.update(loss.item(), input.size(0))
            accr.update(_accr.item(), input.size(0))

            logging.info('valid mini-batch %03d, loss=%e accuracy=%f', step,
                         objs.avg, accr.avg)

    return accr.avg, objs.avg


def main(args):
    benchmark_dictionary = {}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")      
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = utils.benchmark_dataloader(args, kwargs)
    benchmark_file_name = args.benchmark_file_name
    while True:
        logging.info("args = %s", args)
        criterion = nn.CrossEntropyLoss().to(device)
        model = Network(device, nodes=2).to(device)
        with open("all_archs.pkl","rb") as f:
            all_archs = pickle.load(f)
        all_archs = all_archs[args.start_index:args.end_index]
        for arch in all_archs:
            genotype = model.genotype(list(arch))
            if str(genotype) not in benchmark_dictionary:
              test_acc_list = []
              train_acc_list = []
              num_params_list = []
              latency_gpu_mean_list = []
              latency_gpu_std_list = []
              latency_cpu_mean_list = []
              latency_cpu_std_list = []
              for seed in [777]:
                torch.manual_seed(seed)
                model = Network(device, nodes=2).to(device)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
                model.init_arch_params(arch[0], arch[1])
                print(model.genotype())
                if len(benchmark_dictionary.keys()) == 0:
                    #warmup gpu
                    num_params_curr_arch = model.get_num_params(arch)
                num_params_curr_arch = model.get_num_params(arch)
                latency_gpu_mean, latency_gpu_std, latency_cpu_mean, latency_cpu_std, unit_gpu, unit_cpu= model.get_latency()
                logging.info("latency_gpu_mean = %f", latency_gpu_mean)
                logging.info("latency_gpu_std = %f", latency_gpu_std)
                logging.info("latency_cpu_mean = %f", latency_cpu_mean)
                logging.info("latency_cpu_std = %f", latency_cpu_std)
                logging.info("unit_gpu = " + str(unit_gpu))
                logging.info("unit_cpu = " + str(unit_cpu))
                logging.info("param size = %fMB", num_params_curr_arch / 1e6)
                num_params_list.append(num_params_curr_arch)
                latency_gpu_mean_list.append(latency_gpu_mean)
                latency_gpu_std_list.append(latency_gpu_std)
                latency_cpu_mean_list.append(latency_cpu_mean)
                latency_cpu_std_list.append(latency_cpu_std)
                model = model.to(device)
                for epoch in range(args.epochs):
                    logging.info("Starting epoch %d/%d", epoch + 1, args.epochs)
                    # training
                    train_acc, train_obj = train(train_loader, model,
                                     criterion, optimizer, device)
                    logging.info('train_acc %f', train_acc)
                    # validation
                    test_acc, test_obj = infer(test_loader, model, criterion, device)
                    logging.info('test_acc %f', test_acc)
                test_acc_list.append(test_acc)
                train_acc_list.append(train_acc)
                logging.info('test_acc %f', test_acc)
                logging.info('genotype = %s', genotype)
              benchmark_dictionary[str(genotype)] = {"test_acc":np.mean(test_acc_list), "train_acc":np.mean(train_acc_list), "num_params":np.mean(num_params_list), "latency_gpu_mean":np.mean(latency_gpu_mean_list), "latency_gpu_std":np.mean(latency_gpu_std_list), "latency_cpu_mean":np.mean(latency_cpu_mean_list), "latency_cpu_std":np.mean(latency_cpu_std_list), "unit_gpu":unit_gpu, "unit_cpu":unit_cpu}
              with open(benchmark_file_name, 'wb') as f:
                pickle.dump(benchmark_dictionary, f)
        if len(benchmark_dictionary) == 100:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--save', type=str, default='logs', help='path to logs')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--benchmark_file_name', type=str, default='benchmark_dictionary_1.pkl', help='benchmark_file_name')
    parser.add_argument('--start_index', type=int, default=0, help='start_index')
    parser.add_argument('--end_index', type=int, default=100, help='end_index')
    args = parser.parse_args()

    # logging utilities
    os.makedirs(args.save, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)
