import torch
import torch.nn as nn
import torch.nn.functional as F


def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def configure_optimizer(optimizer_old, optimizer_new):
    for i, p in enumerate(optimizer_new.param_groups[0]['params']):
        if not hasattr(p, 'raw_id'):
            optimizer_new.state[p] = optimizer_old.state[p]
            continue
        state_old = optimizer_old.state_dict()['state'][p.raw_id]
        state_new = optimizer_new.state[p]

        state_new['momentum_buffer'] = state_old['momentum_buffer']
        if p.t == 'bn':
            # BN layer
            state_new['momentum_buffer'] = torch.cat([
                state_new['momentum_buffer'],
                state_new['momentum_buffer'][p.out_index].clone()
            ],
                                                     dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id, p.out_index

        elif p.t == 'conv':
            # conv layer
            if hasattr(p, 'in_index'):
                state_new['momentum_buffer'] = torch.cat([
                    state_new['momentum_buffer'],
                    state_new['momentum_buffer'][:, p.in_index, :, :].clone()
                ],
                                                         dim=1)
            if hasattr(p, 'out_index'):
                state_new['momentum_buffer'] = torch.cat([
                    state_new['momentum_buffer'],
                    state_new['momentum_buffer'][p.out_index, :, :, :].clone()
                ],
                                                         dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id
            if hasattr(p, 'in_index'):
                del p.in_index
            if hasattr(p, 'out_index'):
                del p.out_index
    print('%d momemtum buffers loaded' % (i + 1))
    return optimizer_new


def configure_scheduler(scheduler_old, scheduler_new):
    scheduler_new.load_state_dict(scheduler_old.state_dict())
    print('scheduler loaded')
    return scheduler_new
