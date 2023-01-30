import numpy as np  # to handle matrix and data operation
import pandas as pd  # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.7, ), (0.7, )),
])

mnist_train = datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transforms)
train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    mnist_train, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=64,
                                         shuffle=True)
mnist_test = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transforms)
test_loader = torch.utils.data.DataLoader(mnist_test,
                                          batch_size=64,
                                          shuffle=True)

from tiny_model import NetDiscrete

dict_test_accs = {}

for idx1 in range(15):
    for idx2 in range(14):
        model = NetDiscrete(4, 4, 20, idx1, idx2).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        batch_size = 64
        train_errors = []
        train_acc = []
        val_errors = []
        val_acc = []
        n_train = len(train_loader) * batch_size
        n_val = len(val_loader) * batch_size
        for i in range(100):
            total_loss = 0
            total_acc = 0
            c = 0
            for (images, labels) in train_loader:
                images = images.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += torch.sum(
                    torch.max(output, dim=1)[1] == labels).item() * 1.0
                c += 1
            #validation
            total_loss_val = 0
            total_acc_val = 0
            c = 0
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()
                output = model(images)
                loss = criterion(output, labels)
                total_loss_val += loss.item()
                total_acc_val += torch.sum(
                    torch.max(output, dim=1)[1] == labels).item() * 1.0
                c += 1
            train_errors.append(total_loss / n_train)
            train_acc.append(total_acc / n_train)
            val_errors.append(total_loss_val / n_val)
            val_acc.append(total_acc_val / n_val)
            print("Train acc", train_acc[-1])
            print("Val acc", val_acc[-1])
        print("Training complete")
        print(idx1, idx2)
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(train_errors, 'r', label="Train")
        ax[0].plot(val_errors, 'g', label="Validation")
        ax[0].set_title("Error plot")
        ax[0].set_ylabel("Error (cross-entropy)")
        ax[0].set_xlabel("Epoch")
        ax[0].legend()
        ax[1].plot(train_acc, 'r', label="Train")
        ax[1].plot(val_acc, 'g', label="Validation")
        ax[1].set_title("Accuracy plot")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()
        fig.savefig(str(idx1) + str(idx2) + '.png', dpi=fig.dpi)
        total_acc = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            total_acc += torch.sum(
                torch.max(output, dim=1)[1] == labels).item() * 1.0
        print("Test accuracy :", total_acc / len(test_loader.dataset))
        test_acc = total_acc / len(test_loader.dataset)
print(dict_test_accs)
