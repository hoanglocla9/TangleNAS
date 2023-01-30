import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from architect import Architect
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import pandas as pd
'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt',
                        type=str,
                        default=json_path,
                        help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items()
                     if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(
            logger_name, os.path.join(opt['path']['log'],
                                      logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    scheme = opt["train"]["scheme"]
    optimizer_type = opt["netG"]["os_optim"]
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(
                math.ceil(
                    len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info(
                    'Number of train images: {:,d}, iters: {:,d}'.format(
                        len(train_set), train_size))
            if opt['dist']:
                if scheme == "alternative":
                    dataset_train, dataset_val = torch.utils.data.random_split(
                        train_set,
                        [int(0.8 * len(train_set)),
                         int(0.2 * len(train_set))])
                    train_sampler = DistributedSampler(
                        dataset_train,
                        shuffle=dataset_opt['dataloader_shuffle'],
                        drop_last=True,
                        seed=seed)
                    val_sampler = DistributedSampler(
                        dataset_val,
                        shuffle=dataset_opt['dataloader_shuffle'],
                        drop_last=True,
                        seed=seed)
                    train_loader = DataLoader(
                        dataset_train,
                        batch_size=dataset_opt['dataloader_batch_size'] //
                        opt['num_gpu'],
                        shuffle=False,
                        num_workers=dataset_opt['dataloader_num_workers'] //
                        opt['num_gpu'],
                        drop_last=True,
                        pin_memory=True,
                        sampler=train_sampler)
                    val_loader = DataLoader(
                        dataset_val,
                        batch_size=dataset_opt['dataloader_batch_size'] //
                        opt['num_gpu'],
                        shuffle=False,
                        num_workers=dataset_opt['dataloader_num_workers'] //
                        opt['num_gpu'],
                        drop_last=True,
                        pin_memory=True,
                        sampler=val_sampler)
                elif scheme == "simultaneous":
                    #dataset_train, dataset_val = torch.utils.data.random_split(train_set, [int(0.5*len(train_set)),int(0.5*len(train_set))])
                    train_sampler = DistributedSampler(
                        train_set,
                        shuffle=dataset_opt['dataloader_shuffle'],
                        drop_last=True,
                        seed=seed)
                    #val_sampler = DistributedSampler(dataset_val, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                    train_loader = DataLoader(
                        train_set,
                        batch_size=dataset_opt['dataloader_batch_size'] //
                        opt['num_gpu'],
                        shuffle=False,
                        num_workers=dataset_opt['dataloader_num_workers'] //
                        opt['num_gpu'],
                        drop_last=True,
                        pin_memory=True,
                        sampler=train_sampler)
                    #val_loader = DataLoader(dataset_val,
                    #                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                    #                          shuffle=False,
                    #                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                    #                          drop_last=True,
                    #                          pin_memory=True,
                    #                          sampler=val_sampler)

            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt['dataloader_batch_size'],
                    shuffle=dataset_opt['dataloader_shuffle'],
                    num_workers=dataset_opt['dataloader_num_workers'],
                    drop_last=True,
                    pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=False,
                                     pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    columns = [
        'embed_dim',
        'num_rstb',
    ]
    for i in range(4):
        columns.append("num_swin_" + str(i))
        for j in range(6):
            columns.append("num_heads_" + str(i) + "_" + str(j))
            columns.append("mlp_ratio_" + str(i) + "_" + str(j))
    df_archs = pd.DataFrame(columns=columns)
    model = define_Model(opt)
    architect = Architect(model, opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    tau_curr = torch.Tensor([opt["netG"]["tau_max"]])
    prev_psnr = 0
    tau_step = (opt["netG"]["tau_min"] - opt["netG"]["tau_max"]) / 1000
    model.netG.module.sampler.set_taus(opt["netG"]["tau_min"],
                                       opt["netG"]["tau_max"])
    model.netG.module.sampler.set_total_epochs(1000)
    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)
            if scheme == "alternative":
                val_sampler.set_epoch(epoch)
        model.netG.module.sampler.before_epoch()
        if optimizer_type == 'spos':
            arch_params = model.netG.module.sampler.sample_epoch(
                model.netG.module._arch_parameters)
            #print("arch params", arch_params)
        else:
            arch_params = None
        for i, train_data in enumerate(train_loader):

            current_step += 1
            #if scheme == "alternative":
            val_data = next(iter(val_loader))
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            if optimizer_type != "spos":
                architect.step(tau_curr, opt, epoch, val_data)
            #elif scheme == "simultaneous":
            #     architect.optimizer.zero_grad()

            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step, tau_curr, arch_params)
            #if scheme == "simultaneous":
            #    architect.optimizer.step()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt[
                    'rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt[
                    'rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt[
                    'rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test(tau_curr)

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(
                        img_dir,
                        '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img,
                                                       H_img,
                                                       border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(
                        idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.
                    format(epoch, current_step, avg_psnr))
                if avg_psnr > prev_psnr and opt['rank'] == 0:
                    config = model.netG.module.get_best_config()
                    df_archs = df_archs.append(config, ignore_index=True)
                    #print(df_archs.iloc[-1])
                    df_archs.to_pickle("superresolution/" + opt["task"] +
                                       '/df_archs.pkl')
                if avg_psnr > prev_psnr:
                    prev_psnr = avg_psnr

        tau_curr += tau_step


if __name__ == '__main__':
    main()
