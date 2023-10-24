from optimizers.sampler.base_sampler import Sampler
import torch
import numpy as np


class SPOSSampler(Sampler):

    def sample_epoch(self, alphas_list, sample_subset=False):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha, sample_subset))
        return sampled_alphas_list

    def sample_step(self, alphas_list, sample_subset=False):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha, sample_subset))
        return sampled_alphas_list

    def sample_indices(self, num_steps, num_selected):
        indices_to_sample = []
        start = 0
        n = 2
        while True:
            end = start+n
            if end > num_steps:
                break
            choices = np.random.choice([i for i in range(start, end)], num_selected, replace=False)
            for c in list(choices):
                indices_to_sample.append(c)
            start = end
            n = n+1
        return indices_to_sample
        
    def sample(self, alpha, sample_subset=False):
        shape = alpha.shape
        indices_tensor = torch.zeros_like(alpha)
        if len(shape) == 1:
            choice = np.random.choice(alpha.shape[0], 1)[0]
            indices_tensor[choice] = 1
        elif len(shape) == 2:
            if sample_subset==True:
                indices_sampled = self.sample_indices(alpha.shape[0],2)
                for i in indices_sampled:
                    choice = np.random.choice(alpha.shape[-1], 1)[0]
                    indices_tensor[i, choice] = 1  
            else:             
                for i in range(alpha.shape[0]):
                    choice = np.random.choice(alpha.shape[-1], 1)[0]
                    indices_tensor[i, choice] = 1
        elif len(shape) == 3:
            for i in range(alpha.shape[0]):
                for j in range(alpha.shape[1]):
                    choice = np.random.choice(alpha.shape[-1], 1)[0]
                    indices_tensor[i, j, choice] = 1
        return indices_tensor

# test spos
'''if __name__ == '__main__':
    alphas = torch.randn([14,8])
    sampler = SPOSSampler()
    sampled_alphas = sampler.sample(alphas, sample_subset=True)
    print(sampled_alphas)'''