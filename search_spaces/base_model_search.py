from abc import ABC, abstractmethod
import torch
from search_spaces.base_model import NetworkBase


class SearchNetworkBase(NetworkBase):

    @abstractmethod
    def show_alphas(self):
        pass

    @abstractmethod
    def new(self):
        pass

    @abstractmethod
    def _loss(self, input, target):
        pass

    def get_saved_stats(self):
        return {}

    @property
    def is_architect_step(self):
        pass

    @is_architect_step.setter
    def is_architect_step(self, value):
        pass