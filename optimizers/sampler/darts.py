from optimizers.sampler.base_sampler import Sampler


class DARTSSampler(Sampler):

    def sample_epoch(self, alphas_list, sample_subset):
        pass

    def sample_step(self, alphas_list):
        return alphas_list

    def sample(self, alpha):
        return alpha
