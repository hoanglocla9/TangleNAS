
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pickle
import argparse

def get_data(train_portion=0.5):
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    # split trainset into train and valid
    train_len = int(50000*train_portion)
    valid_len = 50000 - train_len
    trainset, validset = torch.utils.data.random_split(trainset, [train_len, valid_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    return trainloader, validloader, testloader

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(7,7),stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(7,7),padding=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(7,7),padding=3,stride=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(7,7),padding=3,stride=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(7,7),stride=1, padding=2)

        self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64,out_features=10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        
    def random_sample_channel(self):
        choices = []
        choices.append(np.random.choice([8,16,32]))
        choices.append(np.random.choice([16,32,64]))
        choices.append(np.random.choice([32,64,128]))
        choices.append(np.random.choice([64,128,256]))
        return choices
    
    def random_sample_kernel(self):
        choices = []
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        return choices
    
    def sample_all_choices(self):
        arch_choices = []
        for c4 in [64,128,256]:
            for k4 in [3,5,7]:
                for c3 in [32,64,128]:
                    for k3 in [3,5,7]:
                        for c2 in [16,32,64]:
                            for k2 in [3,5,7]:
                                for c1 in [8,16,32]:
                                    for k1 in [3,5,7]:
                                        arch_choices.append([[c1,c2,c3,c4],[k1,k2,k3,k4]])
        return arch_choices
    
    def forward(self,x, choices):
        choices_channel  = choices[1]
        choices_kernel = choices[0]
        if choices_kernel[0] == 7:
            x = torch.nn.functional.conv2d(x, self.conv1.weight[:choices_channel[0],:3,:,:], self.conv1.bias[:choices_channel[0]], padding=3)
        elif choices_kernel[0] == 5:
            x = torch.nn.functional.conv2d(x, self.conv1.weight[:choices_channel[0],:3,1:6,1:6], self.conv1.bias[:choices_channel[0]], padding=2)
        elif choices_kernel[0] == 3:
            x = torch.nn.functional.conv2d(x, self.conv1.weight[:choices_channel[0],:3,2:5,2:5], self.conv1.bias[:choices_channel[0]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        if choices_kernel[1] == 7:
            x = torch.nn.functional.conv2d(x, self.conv2.weight[:choices_channel[1],:choices_channel[0],:,:], self.conv2.bias[:choices_channel[1]], padding=3)
        elif choices_kernel[1] == 5:
            x = torch.nn.functional.conv2d(x, self.conv2.weight[:choices_channel[1],:choices_channel[0],1:6,1:6], self.conv2.bias[:choices_channel[1]], padding=2)
        elif choices_kernel[1] == 3:
            x = torch.nn.functional.conv2d(x, self.conv2.weight[:choices_channel[1],:choices_channel[0],2:5,2:5], self.conv2.bias[:choices_channel[1]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        x = self.max_pool(x)
        if choices_kernel[2] == 7:
            x = torch.nn.functional.conv2d(x, self.conv3.weight[:choices_channel[2],:choices_channel[1],:,:], self.conv3.bias[:choices_channel[2]], padding=3)
        elif choices_kernel[2] == 5:
            x = torch.nn.functional.conv2d(x, self.conv3.weight[:choices_channel[2],:choices_channel[1],1:6,1:6], self.conv3.bias[:choices_channel[2]], padding=2)
        elif choices_kernel[2] == 3:
            x = torch.nn.functional.conv2d(x, self.conv3.weight[:choices_channel[2],:choices_channel[1],2:5,2:5], self.conv3.bias[:choices_channel[2]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        x = self.dropout(x)
       
        #print(x.shape)
        x = F.relu(x)
        x = self.max_pool(x)
        if choices_kernel[3] == 7:
            x = torch.nn.functional.conv2d(x, self.conv4.weight[:choices_channel[3],:choices_channel[2],:,:], self.conv4.bias[:choices_channel[3]], padding=3)
        elif choices_kernel[3] == 5:
            x = torch.nn.functional.conv2d(x, self.conv4.weight[:choices_channel[3],:choices_channel[2],1:6,1:6], self.conv4.bias[:choices_channel[3]], padding=2)
        elif choices_kernel[3] == 3:
            x = torch.nn.functional.conv2d(x, self.conv4.weight[:choices_channel[3],:choices_channel[2],2:5,2:5], self.conv4.bias[:choices_channel[3]], padding=1)
        #print(x.shape)
        x = F.relu(x)
        x = torch.nn.functional.conv2d(x, self.conv5.weight[:,:choices_channel[3],:,:], self.conv5.bias, padding=2)
        #print(x.shape)

        x = self.dropout(x)
        x = x.view(-1,6*6*256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return logits
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=3e-4,betas=(0.9,0.995),weight_decay=5e-4)
        return optimizer
    
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
            outputs = model(images, config)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def train_and_eval(self, kernels, channels, model, portion=1.0):
        """
        Function that computes the error on the validation split. 
        Since every architecture has already been trained partially, 
        we just to forward props on the pre-trained supernet 
        """

        #print(arch_params)
        # TODO: check if config is better than current incumbent
        config = (kernels, channels)
        valid_acc = self.eval(config, self.valid_loader, model)
        test_acc = self.eval(config, self.test_loader, model)
        print(str(config))

        # If we find a better validation error, we update the incumbent, else revet to the best current incumbent
        if max(self.curr_incumbent_valid_acc, valid_acc) == valid_acc:
            self.curr_incumbent_valid_acc = valid_acc
            self.curr_incumbent_test_acc = test_acc
            self.curr_incumbent = config
            self.incumbent_trajectory.append(config)
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
        self.model = ConvNet()
        model_state_dict = torch.load(model_path)
        state_dict = {}
        for key in model_state_dict.keys():
            if "arch" not in key:
                state_dict[key] = model_state_dict[key]
        self.model.load_state_dict(state_dict)

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
            kernels = self.model.random_sample_kernel()
            channels = self.model.random_sample_channel()
            self.train_and_eval(kernels,channels,self.model, self.train_portion)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/work/dlclarge2/sukthank-tanglenas/merge/TangleNAS-dev/out_train_spos_spos_9004_0.8/ckpt.pt")
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--train_portion', type=float, default=0.8)
    args = parser.parse_args()
    exp_name = args.model_path.split("/")[-2]
    trainloader, validloader, testloader = get_data(args.train_portion)
    rs = RandomSearch(args.model_path,  trainloader, validloader, testloader, args.train_portion, exp_name)
    rs.optimize(args.n_iters)