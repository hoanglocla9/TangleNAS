import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=(3,3),stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=(3,3),stride=1, padding=0)

        self.fc1 = nn.Linear(in_features=6*6*256,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64,out_features=10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.features = {}

        
    def forward(self,x, get_features=False):
        x = self.conv1(x)
        if get_features:
            x_shape = x.shape
            if "layer1" not in self.features:
                self.features["layer1"] = []
            self.features["layer1"].append(x.reshape(x.shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer2" not in self.features:
                self.features["layer2"] = []
            self.features["layer2"].append(x.reshape(x_shape[0], -1).detach())
        x = self.conv2(x)
        if get_features:
            if "layer3" not in self.features:
                self.features["layer3"] = []
            self.features["layer3"].append(x.reshape(x_shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer4" not in self.features:
                self.features["layer4"] = []
            self.features["layer4"].append(x.reshape(x_shape[0], -1).detach())
        x = self.max_pool(x)
        if get_features:
            if "layer5" not in self.features:
                self.features["layer5"] = []
            self.features["layer5"].append(x.reshape(x_shape[0], -1).detach())
        x = self.conv3(x)
        if get_features:
            if "layer6" not in self.features:
                self.features["layer6"] = []
            self.features["layer6"].append(x.reshape(x_shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer7" not in self.features:
                self.features["layer7"] = []
            self.features["layer7"].append(x.reshape(x_shape[0], -1).detach())
        x = self.dropout(x)
        if get_features:
            if "layer8" not in self.features:
                self.features["layer8"] = []
            self.features["layer8"].append(x.reshape(x_shape[0], -1).detach())
        x = self.conv4(x)
        if get_features:
            if "layer9" not in self.features:
                self.features["layer9"] = []
            self.features["layer9"].append(x.reshape(x_shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer10" not in self.features:
                self.features["layer10"] = []
            self.features["layer10"].append(x.reshape(x_shape[0], -1).detach())
        x = self.max_pool(x)
        if get_features:
            if "layer11" not in self.features:
                self.features["layer11"] = []
            self.features["layer11"].append(x.reshape(x_shape[0], -1).detach())
        x = self.conv5(x)
        if get_features:
            if "layer12" not in self.features:
                self.features["layer12"] = []
            self.features["layer12"].append(x.reshape(x_shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer13" not in self.features:
                self.features["layer13"] = []
            self.features["layer13"].append(x.reshape(x_shape[0], -1).detach())
        x = self.dropout(x)
        if get_features:
            if "layer14" not in self.features:
                self.features["layer14"] = []
            self.features["layer14"].append(x.reshape(x_shape[0], -1).detach())
        x = x.view(-1,6*6*256)
        x = self.fc1(x)
        if get_features:
            if "layer15" not in self.features:
                self.features["layer15"] = []
            self.features["layer15"].append(x.reshape(x_shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer16" not in self.features:
                self.features["layer16"] = []
            self.features["layer16"].append(x.reshape(x_shape[0], -1).detach())
        x = self.fc2(x)
        if get_features:
            if "layer17" not in self.features:
                self.features["layer17"] = []
            self.features["layer17"].append(x.reshape(x_shape[0], -1).detach())
        x = F.relu(x)
        if get_features:
            if "layer18" not in self.features:
                self.features["layer18"] = []
            self.features["layer18"].append(x.reshape(x_shape[0], -1).detach())
        if get_features:
            if "layer20" not in self.features:
                self.features["layer20"] = []
            self.features["layer20"].append(x.reshape(x_shape[0], -1).detach())
        logits = self.fc4(x)
        if get_features:
            if "layer21" not in self.features:
                self.features["layer21"] = []
            self.features["layer21"].append(logits.reshape(x_shape[0], -1).detach())
        return logits
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=3e-4,betas=(0.9,0.995),weight_decay=5e-4)
        return optimizer
    
def train_model(model, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = model.configure_optimizers()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def test(model):
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images, get_features=True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break

    feature_list = []
    for k in model.features.keys():
        model.features[k] = torch.cat(model.features[k], dim=0)
        feature_list.append(model.features[k])
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    return feature_list