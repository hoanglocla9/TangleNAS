import collections
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import argparse

import torch

import contextlib
import pickle
import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import argparse
import os
import yaml
# Encoder: take a string, output a list of integers
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pickle
import argparse

def get_data(train_portion=0.5):
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    # split trainset into train and valid
    train_len = int(50000*train_portion)
    valid_len = 50000 - train_len
    trainset, validset = torch.utils.data.random_split(trainset, [train_len, valid_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    return trainloader, validloader, testloader

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(7,7),stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(7,7),padding=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(7,7),padding=3,stride=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(7,7),padding=3,stride=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(7,7),stride=1, padding=2)

        self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64,out_features=10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        
    def random_sample_channel(self):
        choices = []
        choices.append(np.random.choice([8,16,32]))
        choices.append(np.random.choice([16,32,64]))
        choices.append(np.random.choice([32,64,128]))
        choices.append(np.random.choice([64,128,256]))
        return choices
    
    def random_sample_kernel(self):
        choices = []
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        return choices
    
    def sample_all_choices(self):
        arch_choices = []
        for c4 in [64,128,256]:
            for k4 in [3,5,7]:
                for c3 in [32,64,128]:
                    for k3 in [3,5,7]:
                        for c2 in [16,32,64]:
                            for k2 in [3,5,7]:
                                for c1 in [8,16,32]:
                                    for k1 in [3,5,7]:
                                        arch_choices.append([[c1,c2,c3,c4],[k1,k2,k3,k4]])
        return arch_choices
    
    def forward(self,x, choices):
        choices_channel  = [choices[0],choices[1],choices[2],choices[3]]
        choices_kernel = [choices[4],choices[5],choices[6],choices[7]]
        if choices_kernel[0] == 7:
            x = torch.nn.functional.conv2d(x, self.conv1.weight[:choices_channel[0],:3,:,:], self.conv1.bias[:choices_channel[0]], padding=3)
        elif choices_kernel[0] == 5:
            x = torch.nn.functional.conv2d(x, self.conv1.weight[:choices_channel[0],:3,1:6,1:6], self.conv1.bias[:choices_channel[0]], padding=2)
        elif choices_kernel[0] == 3:
            x = torch.nn.functional.conv2d(x, self.conv1.weight[:choices_channel[0],:3,2:5,2:5], self.conv1.bias[:choices_channel[0]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        if choices_kernel[1] == 7:
            x = torch.nn.functional.conv2d(x, self.conv2.weight[:choices_channel[1],:choices_channel[0],:,:], self.conv2.bias[:choices_channel[1]], padding=3)
        elif choices_kernel[1] == 5:
            x = torch.nn.functional.conv2d(x, self.conv2.weight[:choices_channel[1],:choices_channel[0],1:6,1:6], self.conv2.bias[:choices_channel[1]], padding=2)
        elif choices_kernel[1] == 3:
            x = torch.nn.functional.conv2d(x, self.conv2.weight[:choices_channel[1],:choices_channel[0],2:5,2:5], self.conv2.bias[:choices_channel[1]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        x = self.max_pool(x)
        if choices_kernel[2] == 7:
            x = torch.nn.functional.conv2d(x, self.conv3.weight[:choices_channel[2],:choices_channel[1],:,:], self.conv3.bias[:choices_channel[2]], padding=3)
        elif choices_kernel[2] == 5:
            x = torch.nn.functional.conv2d(x, self.conv3.weight[:choices_channel[2],:choices_channel[1],1:6,1:6], self.conv3.bias[:choices_channel[2]], padding=2)
        elif choices_kernel[2] == 3:
            x = torch.nn.functional.conv2d(x, self.conv3.weight[:choices_channel[2],:choices_channel[1],2:5,2:5], self.conv3.bias[:choices_channel[2]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        x = self.dropout(x)
       
        #print(x.shape)
        x = F.relu(x)
        x = self.max_pool(x)
        if choices_kernel[3] == 7:
            x = torch.nn.functional.conv2d(x, self.conv4.weight[:choices_channel[3],:choices_channel[2],:,:], self.conv4.bias[:choices_channel[3]], padding=3)
        elif choices_kernel[3] == 5:
            x = torch.nn.functional.conv2d(x, self.conv4.weight[:choices_channel[3],:choices_channel[2],1:6,1:6], self.conv4.bias[:choices_channel[3]], padding=2)
        elif choices_kernel[3] == 3:
            x = torch.nn.functional.conv2d(x, self.conv4.weight[:choices_channel[3],:choices_channel[2],2:5,2:5], self.conv4.bias[:choices_channel[3]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        x = torch.nn.functional.conv2d(x, self.conv5.weight[:,:choices_channel[3],:,:], self.conv5.bias, padding=2)
        #print(x.shape)

        x = self.dropout(x)
        x = x.view(-1,6*6*256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return logits
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=3e-4,betas=(0.9,0.995),weight_decay=5e-4)
        return optimizer
#print(len(eval_data))
def decode_cand_tuple(cand_tuple):
    channel_size_1 = cand_tuple[0]
    channel_size_2 = cand_tuple[1]
    channel_size_3 = cand_tuple[2]
    channel_size_4 = cand_tuple[3]
    kernel_size_1 = cand_tuple[4]
    kernel_size_2 = cand_tuple[5]
    kernel_size_3 = cand_tuple[6]
    kernel_size_4 = cand_tuple[7]
    return kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, channel_size_1, channel_size_2, channel_size_3, channel_size_4

class EvolutionSearcher(object):

    def __init__(self, args, device, model, model_without_ddp, choices, output_dir, train_loader, valid_loader, test_loader):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.output_dir = output_dir
        self.s_prob =args.s_prob
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.eval_iters = 200
        self.candidates = []
        self.top_losses = []
        self.cand_params = []
        self.choices = choices
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def save_checkpoint(self):

        info = {}
        info['top_accs'] = self.top_losses
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        print(cand)
        channel_size_1, channel_size_2, channel_size_3, channel_size_4, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4 = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config['channel_size_1'] = channel_size_1
        sampled_config['channel_size_2'] = channel_size_2
        sampled_config['channel_size_3'] = channel_size_3
        sampled_config['channel_size_4'] = channel_size_4
        sampled_config['kernel_size_1'] = kernel_size_1
        sampled_config['kernel_size_2'] = kernel_size_2
        sampled_config['kernel_size_3'] = kernel_size_3
        sampled_config['kernel_size_4'] = kernel_size_4
        #print(arch_param)
        eval_acc, test_acc = self.estimate_acc(cand, self.model)

        info['eval_acc'] = eval_acc
        info['test_acc'] = test_acc

        info['visited'] = True

        return True

    def estimate_acc(self, config, model):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
         for data in self.valid_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images, config)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc =  100 * correct / total
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # calculate outputs by running images through the network
                outputs = model(images, config)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc =  100 * correct / total
        return val_acc, test_acc
        
    
    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=100):
        while True:
            cands = []
            for _ in range(batchsize):
                cand = random_func()
                cands.append(cand)
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            return cands

    def get_random_cand(self):
        cand_tuple = list()
        channels = self.model.random_sample_channel()
        kernels = self.model.random_sample_kernel()
        channel_1, channel_2, channel_3, channel_4 = channels
        kernel_1, kernel_2, kernel_3, kernel_4 = kernels
        selected_config = {}
        selected_config['channel_1'] = channel_1
        selected_config['channel_2'] = channel_2
        selected_config['channel_3'] = channel_3
        selected_config['channel_4'] = channel_4
        selected_config['kernel_1'] = kernel_1
        selected_config['kernel_2'] = kernel_2
        selected_config['kernel_3'] = kernel_3
        selected_config['kernel_4'] = kernel_4
        for k in selected_config.keys():
            cand_tuple.append(selected_config[k])

        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        i = 0
        while len(self.candidates) < num:
            cand = cand_iter[i]

            if not self.is_legal(cand):
                i += 1
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
            i += 1
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        res_arch = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            # Choose a random candidate
            cand = list(random.choice(self.keep_top_k[k]))
            channel_1, channel_2, channel_3, channel_4 = cand[0], cand[1], cand[2], cand[3]
            kernel_1, kernel_2, kernel_3, kernel_4 = cand[4], cand[5], cand[6], cand[7]
            random_s = random.random()
            new_depth = None
            # depth

            if random_s < m_prob: # check is depth is mutated
               new_channel_1 = random.choice(self.choices["channel_size_1"])
            else:
                new_channel_1 = channel_1
            if random.random() < m_prob:
                new_channel_2 = random.choice(self.choices["channel_size_2"])
            else:
                new_channel_2 = channel_2
            if random.random() < m_prob:
                new_channel_3 = random.choice(self.choices["channel_size_3"])
            else:
                new_channel_3 = channel_3
            if random.random() < m_prob:
                new_channel_4 = random.choice(self.choices["channel_size_4"])
            else:
                new_channel_4 = channel_4
            if random.random() < m_prob:
                new_kernel_1 = random.choice(self.choices["kernel_size_1"])
            else:
                new_kernel_1 = kernel_1
            if random.random() < m_prob:
                new_kernel_2 = random.choice(self.choices["kernel_size_2"])
            else:
                new_kernel_2 = kernel_2
            if random.random() < m_prob:
                new_kernel_3 = random.choice(self.choices["kernel_size_3"])   
            else:
                new_kernel_3 = kernel_3
            if random.random() < m_prob:
                new_kernel_4 = random.choice(self.choices["kernel_size_4"])    
            else:
                new_kernel_4 = kernel_4       
            # mutated candidate
            result_cand = [new_channel_1, new_channel_2, new_channel_3, new_channel_4, new_kernel_1, new_kernel_2, new_kernel_3, new_kernel_4]
            output = tuple(result_cand)
            return output

        cand_iter = self.stack_random_cand(random_func)
        i = 0
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = cand_iter[i]
            if not self.is_legal(cand):
                i += 1
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))
            i += 1

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res_cand = []
        max_iters = 10 * crossover_num

        def random_func():

            # choose parent 1 (p1) and parent 2 (p2) randomly from top k
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            # randomly choose from p1 and p2 to contruct child
            channel_1 = random.choice([p1[0], p2[0]])
            channel_2 = random.choice([p1[1], p2[1]])
            channel_3 = random.choice([p1[2], p2[2]])
            channel_4 = random.choice([p1[3], p2[3]])
            kernel_1 = random.choice([p1[4], p2[4]])
            kernel_2 = random.choice([p1[5], p2[5]])
            kernel_3 = random.choice([p1[6], p2[6]])
            kernel_4 = random.choice([p1[7], p2[7]])
            # crossover candidate
            result_cand = [channel_1, channel_2, channel_3, channel_4, kernel_1, kernel_2, kernel_3, kernel_4]
            output = tuple(result_cand)
            return output

        cand_iter = self.stack_random_cand(random_func)
        i = 0
        while len(res_cand) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = cand_iter[i]
            if not self.is_legal(cand):
                i  = i +1 
                continue
            res_cand.append(cand)
            print('crossover {}/{}'.format(len(res_cand), crossover_num))
            i += 1

        print('crossover_num = {}'.format(len(res_cand)))
        return res_cand

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['eval_acc'],reverse=True)
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['eval_acc'], reverse=True)

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, Top-1 test acc = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['eval_acc'], self.vis_dict[cand]['test_acc'], 0))
                tmp_accuracy.append(self.vis_dict[cand]['eval_acc'])
            self.top_losses.append(tmp_accuracy)

            mutation_cands = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover_cands = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation_cands + crossover_cands

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=30) #30
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=10) # 10 
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--mutation-num', type=int, default=10)
    parser.add_argument('--param-limits', type=float, default=10000000)
    parser.add_argument('--min-param-limits', type=float, default=0)


    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='evo_checkpoint_conv_macro',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.add_argument('--train_portion', default=0.8, type=float, help='portion of training data')
    parser.add_argument('--model_path', default='/path/to/ckpt.pt', type=str, help='path to pretrained model')
    parser.set_defaults(amp=True)

    return parser

def main(args):

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # save config for later experiments
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        f.write(args_text)
    # fix the seed for reproducibility

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    print(f"Creating CharLM")
    model_path = args.model_path
    model_name  = model_path.split("/")[-2]
    args.output_dir = os.path.join(args.output_dir, model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = ConvNet()
    choices = {}
    choices["channel_size_1"] = [8,16,32]
    choices["channel_size_2"] = [16,32,64]
    choices["channel_size_3"] = [32,64,128]
    choices["channel_size_4"] = [64,128,256]
    choices["kernel_size_1"] = [3,5,7]
    choices["kernel_size_2"] = [3,5,7]
    choices["kernel_size_3"] = [3,5,7]
    choices["kernel_size_4"] = [3,5,7]
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model_without_ddp = model


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    t = time.time()
    train_loader, val_loader, test_loader = get_data(args.train_portion)
    searcher = EvolutionSearcher(args, device, model, model_without_ddp, choices, args.output_dir, train_loader, val_loader, test_loader)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conv Macro evolution search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)