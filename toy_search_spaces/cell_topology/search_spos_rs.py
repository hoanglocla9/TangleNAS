import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pickle
import argparse
from toy_search_spaces.cell_topology.model_search_temp import ToyCellSearchSpace
from toy_search_spaces.cell_topology.utils import spos_search_dataloader
class NASOptimizer(object):
    """
    Base class for NASBench-101 optimizers. All subclasses should
    inherit from this.
    """

    def __init__(self):
        # get the configuration space
        # configuration (architecture) at each point in time.
        # incumbent_trajectory_error keeps track of the
        # corresponding validation errors of incumbent_trajectory
        self.incumbent_trajectory = []
        self.incumbent_trajectory_valid_acc = []
        self.incumbent_trajectory_test_acc = []
        self.curr_wallclock = 0
        self.curr_incumbent = None
        self.curr_incumbent_valid_acc= -10000000
        self.eval_iters = 200

    def optimize(self, n_iters: int = 100):
        raise NotImplementedError
    
    def eval(self,config, dataloader, model):
        """
        Function that computes the error on the validation split. 
        Since every architecture has already been trained partially, 
        we just do forward props on the pre-trained supernet 
        """
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
         for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # calculate outputs by running images through the network
            _, outputs = model(images, alphas=config)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def train_and_eval(self, alpha_sampled, model, portion=1.0):
        """
        Function that computes the error on the validation split. 
        Since every architecture has already been trained partially, 
        we just to forward props on the pre-trained supernet 
        """

        #print(arch_params)
        # TODO: check if config is better than current incumbent
        config = alpha_sampled
        valid_acc = self.eval(config, self.valid_loader, model)
        test_acc = self.eval(config, self.test_loader, model)
        genotype = model.genotype(alphas=alpha_sampled)

        # If we find a better validation error, we update the incumbent, else revet to the best current incumbent
        if max(self.curr_incumbent_valid_acc, valid_acc) == valid_acc:
            self.curr_incumbent_valid_acc = valid_acc
            self.curr_incumbent_test_acc = test_acc
            self.curr_incumbent = genotype
            self.incumbent_trajectory.append(genotype)
            self.incumbent_trajectory_valid_acc.append(valid_acc)
            self.incumbent_trajectory_test_acc.append(test_acc)
        else:
            self.incumbent_trajectory.append(self.curr_incumbent)
            self.incumbent_trajectory_valid_acc.append(
                valid_acc)
            self.incumbent_trajectory_test_acc.append(
                test_acc)
        print("Current incumbent error: ", self.curr_incumbent_valid_acc)
        print("Current incumbent test error: ", self.curr_incumbent_test_acc)
        print("Current incumbent: ", self.curr_incumbent)
        incumbent_valid_file = "incumbent_trajectory_valid_acc_rs_{}_{}.pkl".format(portion,self.exp_name)
        incumbent_test_file = "incumbent_trajectory_test_acc_rs_{}_{}.pkl".format(portion, self.exp_name)
        incumbent_file = "incumbent_trajectory_rs_{}_{}.pkl".format(portion,self.exp_name)
        with open(incumbent_valid_file, "wb") as f:
            pickle.dump(self.incumbent_trajectory_valid_acc, f)
        with open(incumbent_test_file, "wb") as f:
            pickle.dump(self.incumbent_trajectory_test_acc, f)
        with open(incumbent_file, "wb") as f:
            pickle.dump(self.incumbent_trajectory, f)


class RandomSearch(NASOptimizer):
    """
    Algorithm for random search.
    """

    def __init__(self, model_path, train_data, eval_data, test_data, train_portion=1.0, exp_name="random_search"):
        super(RandomSearch, self).__init__()
        self.model_path = model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ToyCellSearchSpace("spos",10,criterion=torch.nn.CrossEntropyLoss(),entangle_weights=True, use_we_v2=True)
        model_state_dict = torch.load(model_path)["search_model"]
        #print(model_state_dict.keys())
        self.model.load_state_dict(model_state_dict)

        self.model = self.model.to(device)
        self.train_loader = train_data
        self.exp_name = exp_name
        self.valid_loader = eval_data
        self.test_loader = test_data
        self.train_portion = train_portion

    def optimize(self, n_iters: int = 100):
        """
        Run random search for n_iters function evaluations.
        """
        for i in range(n_iters):
            alpha_sampled = self.model.sampler.sample_step(self.model._arch_parameters)
            self.train_and_eval(alpha_sampled,self.model, self.train_portion)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--model_path', type=str, default="/path/to/model.pth")
=======
    parser.add_argument('--model_path', type=str, default="/path/to/ckpt.pt")
>>>>>>> 9f4b5b84d7f3547c835ba81757ef4e286ab2ac3e
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--train_portion', type=float, default=0.8)
    args = parser.parse_args()
    exp_name = args.model_path.split("/")[-2]
    trainloader, validloader, testloader = spos_search_dataloader(train_portion=args.train_portion)
    rs = RandomSearch(args.model_path,  trainloader, validloader, testloader, args.train_portion, exp_name)
    rs.optimize(args.n_iters)