from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

class BaseArchitect(ABC):

    def __init__(self, model, reg_scale=None):

        self.arch_weight_decay = 1e-3
        if reg_scale != None:
            self.arch_weight_decay = reg_scale
        self.lr = 3e-4

        self.model = model
        self.unwrapped_model = model.module if isinstance(self.model, DDP) else model

        self.optimizer = torch.optim.Adam(self.unwrapped_model.get_arch_parameters(),
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

    def __init__(self, model, reg_scale=None):
        super().__init__(model=model, reg_scale=reg_scale)

    def step(self, input_train, target_train, input_valid, target_valid,
             network_optimizer):
        self.optimizer.zero_grad(set_to_none=True)
        logits, loss = self._backward_step(input_valid, target_valid)
        self.optimizer.step()
        return loss, logits

    def _backward_step(self, input_valid, target_valid):
        logits, loss = self.model(input_valid, target_valid)

        loss.backward()
        return loss, logits
