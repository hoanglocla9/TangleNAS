from optimizers.sampler.base_sampler import Sampler
import torch
import torch.nn.functional as F


class DiscreteSampler(Sampler):

    def sample_epoch(self, alphas_list, sample_subset):
        pass

    def sample_step(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha))
        print("Sampled", sampled_alphas_list)
        return sampled_alphas_list

    def sample(self, alpha):
        alpha = torch.nn.functional.softmax(alpha,dim=-1)
        if alpha.shape[1] == 3:
            weights = torch.zeros_like(alpha)
            weights[0,-1]=1
        else:
            weights = torch.zeros_like(alpha)
            for i in range(weights.shape[0]):
                 weights[i,-1]=1
        return weights