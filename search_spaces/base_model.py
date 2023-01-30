from abc import ABC, abstractmethod
import torch


class NetworkBase(ABC, torch.nn.Module):

    def get_weights(self):  # TODO: find a cleaner way to do this
        li = []
        for n, p in self.named_parameters():
            if ("alpha" not in n) and ("arch" not in n):
                li.append(p)
        return li

    def get_named_weights(self):  # TODO: find a cleaner way to do this
        li = {}
        for n, p in self.named_parameters():
            if ("alpha" not in n) and ("arch" not in n):
                li[n] = p
        for k in li.keys():
            yield (k, li[k])

    def arch_parameters(self):
        return self._arch_parameters

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _initialize_alphas(self):
        pass
