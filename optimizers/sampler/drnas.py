from optimizers.sampler.base_sampler import Sampler
import torch
import torch.nn.functional as F


class DRNASSampler(Sampler):

    def sample_epoch(self, alphas_list, sample_subset):
        pass

    def sample_step(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha))
        return sampled_alphas_list

    def sample(self, alpha):
        weights = torch.distributions.dirichlet.Dirichlet(F.elu(alpha.clone(),inplace=False) + 1).rsample()
        return weights
