import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from search_spaces.base_model_search import NetworkBase
from toy_search_spaces.conv_macro.mixture_operations import ConvMixtureFixedInCh, ConvMixtureFixedOutCh, ConvMixture, ConvSuper, ConvSuperFixedInCh, ConvSuperFixedOutCh
from optimizers.optim_factory import get_mixop, get_sampler
import itertools
from optimizers.mixop.entangle import EntangledOp
class ConvNetMacroSpace(NetworkBase):
    
    def __init__(self, optimizer_type="darts_v1", api="/path/to/bench", use_we_v2=False):
        super(ConvNetMacroSpace,self).__init__()
        
        conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(7,7),stride=1, padding=3)
        self.use_we_v2 = use_we_v2
        self.api = api
        self.type = 'toy'
        if use_we_v2:
            self.conv1_mixture = ConvMixtureFixedInCh(conv1, [8,16,32], [3,5,7])
            self.conv1_ops = self.get_entangle_ops_combi(self.conv1_mixture, [8,16,32], [3,5,7], 'conv1')
        else:
            self.conv1_ops = []
            for c in [8,16,32]:
                for k in [3,5,7]:
                    op = ConvSuperFixedInCh(conv1, c, k)
                    self.conv1_ops.append(op)
        conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(7,7),padding=3,stride=1)
        if use_we_v2:
            self.conv2_mixture = ConvMixture(conv2, [8,16,32], [16,32,64], [3,5,7])
            self.conv2_ops = self.get_entangle_ops_combi3(self.conv2_mixture, [8,16,32], [16,32,64], [3,5,7], 'conv2')
        else:
            self.conv2_ops = torch.nn.ModuleList()
            for c1 in [8,16,32]:
                    for c2 in [16,32,64]:
                        for k1 in [3,5,7]:
                            op = ConvSuper(conv2, c1, c2, k1)
                            self.conv2_ops.append(op)
        conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(7,7),padding=3,stride=1)
        if use_we_v2:
            self.conv3_mixture = ConvMixture(conv3, [16,32,64], [32,64,128], [3,5,7])
            self.conv3_ops = self.get_entangle_ops_combi3(self.conv3_mixture, [16,32,64], [32,64,128], [3,5,7], 'conv3')
        else:
            self.conv3_ops = torch.nn.ModuleList()
            for c1 in [16,32,64]:
                    for c2 in [32,64,128]:
                        for k1 in [3,5,7]:
                            op = ConvSuper(conv3, c1, c2, k1)
                            self.conv3_ops.append(op)
        conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(7,7),padding=3,stride=1)
        if use_we_v2:
            self.conv4_mixture = ConvMixture(conv4, [32,64,128], [64,128,256], [3,5,7])
            self.conv4_ops = self.get_entangle_ops_combi3(self.conv4_mixture, [32,64,128], [64,128,256], [3,5,7], 'conv4')
        else:
            self.conv4_ops = torch.nn.ModuleList()
            for c1 in [32,64,128]:
                    for c2 in [64,128,256]:
                        for k1 in [3,5,7]:
                            op = ConvSuper(conv4, c1, c2, k1)
                            self.conv4_ops.append(op)
        conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(7,7),stride=1, padding=2)
        if use_we_v2:
            self.conv5_mixture = ConvMixtureFixedOutCh(conv5, [64,128,256])
            self.conv5_ops = self.get_entangle_ops(self.conv5_mixture, [64,128,256],'conv5')
        else:
            self.conv5_ops = torch.nn.ModuleList()
            for c in [64,128,256]:
                op = ConvSuperFixedOutCh(conv5, c)
                self.conv5_ops.append(op)
        self._criterion = nn.CrossEntropyLoss()
        self.optimizer_type = optimizer_type
        self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64,out_features=10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.mixop = get_mixop(optimizer_type,use_we_v2=use_we_v2)
        self.sampler = get_sampler(optimizer_type)
        self._initialize_alphas()
        if self.api is not None:
            with open(self.api, 'rb') as f:
                self.benchmark = pickle.load(f)

    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]
    
    def get_entangle_ops_combi3(self, op, choices1, choices2, choices3, op_name):
        choices = list(itertools.product(choices1, choices2, choices3))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]
        
    def random_sample_channel(self):
        choices = []
        choices.append(np.random.choice([8,16,32]))
        choices.append(np.random.choice([16,32,64]))
        choices.append(np.random.choice([32,64,128]))
        choices.append(np.random.choice([64,128,256]))
        return choices

    def show_alphas(self):
        print("Channels",F.softmax(self.arch_param_channels, dim=-1))
        print("Kernels",F.softmax(self.arch_param_kernels, dim=-1))
        
    def get_saved_stats(self):
        return {}
    
    def random_sample_kernel(self):
        choices = []
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        choices.append(np.random.choice([3,5,7]))
        return choices
    
    def _loss(self, input, target):
        _, logits = self(input)
        loss = self._criterion(logits, target)
        return loss, logits
    
    def new(self):
        # check if cuda is available and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_new = ConvNetMacroSpace(
            self.optimizer_type,
            use_we_v2=self.use_we_v2,
            api=self.api,
            ).to(device)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model_new
    
    def genotype(self):
        def _parse(weights1, weights2):
            gene = []
            channels_chosen = []
            kernels_chosen = []
            for i in range(4):
                kernel_choices = [3,5,7]
                if i == 0:
                    ch_choices = [8,16,32]
                elif i ==1:
                    ch_choices = [16,32,64]
                elif i ==2:
                    ch_choices = [32,64,128]
                else:
                    ch_choices = [64,128,256]
                ch, op = ch_choices[weights1[i].argmax()], kernel_choices[weights2[i].argmax()]
                channels_chosen.append(ch)
                kernels_chosen.append(op)
            gene=[channels_chosen, kernels_chosen]
            return gene
        return _parse(F.softmax(self.arch_param_channels, dim=-1).data.cpu().numpy(), F.softmax(self.arch_param_kernels, dim=-1).data.cpu().numpy())
    
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
    
    def _initialize_alphas(self):
        self.arch_param_channels = torch.nn.Parameter(1e-3*torch.randn(4,3),requires_grad=True)
        self.arch_param_kernels = torch.nn.Parameter(1e-3*torch.randn(4,3),requires_grad=True)
        self._arch_parameters = [self.arch_param_channels, self.arch_param_kernels]
    
    def forward(self, x, arch_params=None):
        if arch_params is None:
            arch_params_sampled = self.sampler.sample_step(self._arch_parameters)
        else:
            arch_params_sampled = arch_params
        x = self.mixop.forward(x, [arch_params_sampled[0][0], arch_params_sampled[1][0]],self.conv1_ops,  combi=True)
        x = F.relu(x)
        x = self.mixop.forward(x, [arch_params_sampled[0][0], arch_params_sampled[0][1], arch_params_sampled[1][1]],self.conv2_ops, combi=True)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.mixop.forward(x, [arch_params_sampled[0][1], arch_params_sampled[0][2], arch_params_sampled[1][2]], self.conv3_ops, combi=True)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.mixop.forward(x, [arch_params_sampled[0][2], arch_params_sampled[0][3], arch_params_sampled[1][3]], self.conv4_ops, combi=True)
        x = F.relu(x)
        x = self.mixop.forward(x, arch_params_sampled[0][3], self.conv5_ops, combi=False)

        x = self.dropout(x)
        x = x.view(-1,6*6*256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return x, logits
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.get_weights(),lr=3e-4,betas=(0.9,0.995),weight_decay=5e-4)
        return optimizer
    
    def query(self):
        genotype = self.genotype()
        return self.benchmark[str(genotype)]
    

'''# init model
model = ConvNetMacroSpace(optimizer_type="gdas")
# init data
x = torch.randn(1,3,32,32)
# init arch params
model._initialize_alphas()
# forward pass
model.sampler.set_taus(0.1,10)
model.sampler.set_total_epochs(100)
model.sampler.before_epoch()
logits = model(x)
# backward pass
logits[-1].mean().backward()
for p in model.arch_parameters():
    print(p.grad)'''
