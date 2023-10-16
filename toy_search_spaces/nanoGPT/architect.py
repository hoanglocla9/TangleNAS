from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
import numpy as np


class BaseArchitect(ABC):

    def __init__(self, model, reg_scale=None):
        self.network_weight_decay = 3e-4
        self.model = model
        self.arch_weight_decay = 1e-3
        if reg_scale != None:
            self.arch_weight_decay = reg_scale
        self.lr = 3e-4

        self.optimizer = torch.optim.Adam(self.model.module.get_arch_parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.999),
                                          weight_decay=self.arch_weight_decay)
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    @abstractmethod
    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer):
        ...


class ArchitectV1(BaseArchitect):

    def __init__(self, model, use_kl_loss=False, reg_scale=None):
        super().__init__(model=model, reg_scale=reg_scale)
        self.use_kl_loss = use_kl_loss

    def step(self, input_train, target_train, input_valid, target_valid,
             network_optimizer):
        self.optimizer.zero_grad(set_to_none=True)
        logits, loss = self._backward_step(input_valid, target_valid)
        self.optimizer.step()
        return loss, logits

    def _backward_step(self, input_valid, target_valid):
        logits, loss = self.model(input_valid, target_valid)

        if self.use_kl_loss:
            loss += self.model._get_kl_reg()

        loss.backward()
        return loss, logits


class ArchitectV2(BaseArchitect):

    def __init__(self, model, config):
        super().__init__(model=model, config=config)

    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer):

        loss, logits = self._backward_step_unrolled(input_train, target_train,
                                                    input_valid, target_valid,
                                                    eta, network_optimizer)

        self.optimizer.step()
        return loss, logits

    def _concat(self, xs):
        return torch.cat([x.view(-1) for x in xs])

    def _backward_step_unrolled(self, input_train, target_train, input_valid,
                                target_valid, eta, network_optimizer):

        unrolled_model = self._compute_unrolled_model(input_train,
                                                      target_train, eta,
                                                      network_optimizer)

        unrolled_loss, logits = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.get_weights()]
        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        return unrolled_loss, logits

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss, _ = self.model._loss(input, target)
        theta = self._concat(self.model.get_weights()).data
        #loss.backward()
        #for n,p in self.model.get_named_weights():
        #    if p.grad==None:
        #        print(n)
        try:
            moment = self._concat(network_optimizer.state[v]['momentum_buffer']
                                  for v in self.model.get_weights()).mul_(
                                      self.network_momentum)
        except BaseException:
            moment = torch.zeros_like(theta)
        #loss.backward()

        dtheta = self._concat(
            torch.autograd.grad(
                loss, self.model.get_weights(),
                allow_unused=True)).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))

        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.get_named_weights():
            v_length = np.prod(v.size())
            params[k] = theta[offset:offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(self.device)

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / self._concat(vector).norm()

        for p, v in zip(self.model.get_weights(), vector):
            p.data.add_(R, v)
        loss, _ = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss,
                                      self.model.arch_parameters(),
                                      allow_unused=True)

        for p, v in zip(self.model.get_weights(), vector):
            p.data.sub_(2 * R, v)
        loss, _ = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss,
                                      self.model.arch_parameters(),
                                      allow_unused=True)

        for p, v in zip(self.model.get_weights(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


class DummyArchitect(BaseArchitect):

    def __init__(self, model, config):
        super().__init__(model=model, config=config)

    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer):
        return torch.Tensor([-1.]).to(self.device), torch.ones(
            target_valid.shape[0], 5).to(self.device) * -1