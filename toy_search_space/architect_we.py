import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from tiny_model_we import Net


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model):
        self.network_momentum = 0.9
        self.network_weight_decay = 3e-4
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.get_arch_params(),
                                          lr=3e-4,
                                          betas=(0.5, 0.999),
                                          weight_decay=1e-3)
        self.loss = nn.CrossEntropyLoss()

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.loss(self.model(input, 0.1), target)
        theta = _concat(self.model.get_model_params()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.get_arch_params()).mul_(
                                 self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(loss, self.model.get_model_params())
        ).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train,
                                         input_valid, target_valid, eta,
                                         network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.loss(self.model(input_valid, 0.1), target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid,
                                target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train,
                                                      target_train, eta,
                                                      network_optimizer)
        unrolled_loss = self.loss(unrolled_model(input_valid, 0.1),
                                  target_valid)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.get_arch_params()]
        vector = [v.grad.data for v in unrolled_model.get_model_params()]
        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.model.get_arch_params(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = Net(4, 4, 20, 'darts').cuda()
        model_dict = self.model.state_dict()
        params, offset = {}, 0
        for k, v in self.model.get_named_params():
            v_length = np.prod(v.size())
            params[k] = theta[offset:offset + v_length].view(v.size())
            offset += v_length
        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.get_model_params(), vector):
            p.data.add_(R, v)
        loss = self.loss(self.model(input, 0.1), target)
        grads_p = torch.autograd.grad(loss, self.model.get_arch_params())
        for p, v in zip(self.model.get_model_params(), vector):
            p.data.sub_(2 * R, v)
        loss = self.loss(self.model(input, 0.1), target)
        grads_n = torch.autograd.grad(loss, self.model.get_arch_params())
        for p, v in zip(self.model.get_model_params(), vector):
            p.data.add_(R, v)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
