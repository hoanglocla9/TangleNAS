from abc import ABC, abstractmethod
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from .transforms import CUTOUT
from .imgnet16 import ImageNet16


class ExperimentData(ABC):

    def __init__(self, root, train_portion=1.0):
        self.root = root
        self.train_portion = train_portion
        if train_portion == 1:
            self.shuffle = True
        else:
            self.shuffle = False

    @abstractmethod
    def build_datasets(self):
        ...

    @abstractmethod
    def get_transforms(self):
        ...

    @abstractmethod
    def load_datasets(self):
        ...

    def preprocess(self, inputs, labels):
        return inputs, labels

    def get_dataloaders(self, batch_size=64, n_workers=2):
        (train_data, train_sampler), (val_data, val_sampler), test_data = self.build_datasets()
        train_queue = torch.utils.data.DataLoader(train_data,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  sampler=train_sampler,
                                                  num_workers=n_workers,
                                                  shuffle = self.shuffle)
        if val_data != None:
            valid_queue = torch.utils.data.DataLoader(val_data,
                                                      batch_size=batch_size,
                                                      pin_memory=True,
                                                      sampler = val_sampler,
                                                      num_workers=n_workers)
        else:
            valid_queue = None

        test_queue = torch.utils.data.DataLoader(test_data,
                                                 batch_size=batch_size,
                                                 pin_memory=True,
                                                 num_workers=n_workers)

        return train_queue, valid_queue, test_queue


class CIFARData(ExperimentData):

    def __init__(self, root, cutout, train_portion=0.5):
        super().__init__(root, train_portion)
        self.cutout = cutout

    def get_transforms(self):

        lists = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]

        if self.cutout > 0:
            lists += [CUTOUT(self.cutout)]
        train_transform = transforms.Compose(lists)

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(self.mean, self.std)])

        return train_transform, test_transform

    def build_datasets(self):

        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(self.root, train_transform,
                                                   test_transform)

        if self.train_portion < 1:
            num_train = len(train_data)
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])
            return (train_data, train_sampler), (train_data, val_sampler), test_data

        else:
            return (train_data, None), (None, None), test_data


class ImageNetData(ExperimentData):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, train_portion)
        self.cutout = cutout
        self.mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        self.std = [x / 255 for x in [63.22, 61.26, 65.09]]

    def get_transforms(self):
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        if self.cutout > 0:
            lists += [CUTOUT(self.cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(self.mean, self.std)])

        return train_transform, test_transform

    def build_datasets(self):

        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(self.root, train_transform,
                                                   test_transform)

        if self.train_portion > 0:
            num_train = len(train_data)
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])
            return (train_data, train_sampler), (train_data, val_sampler), test_data
        else:
            return (train_data,None), (None, None), test_data


class CIFAR10Data(CIFARData):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)
        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]
        #self.mean = [0.5, 0.5, 0.5]
        #self.std = [0.5, 0.5, 0.5]

    def load_datasets(self, root, train_transform, test_transform):
        train_data = dset.CIFAR10("./data",
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10("./data",
                                 train=False,
                                 transform=test_transform,
                                 download=True)

        assert len(train_data) == 50000 and len(test_data) == 10000
        return train_data, test_data


class CIFAR100Data(CIFARData):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)
        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]

    def load_datasets(self, root, train_transform, test_transform):
        train_data = dset.CIFAR100("./data",
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100("./data",
                                  train=False,
                                  transform=test_transform,
                                  download=True)

        assert len(train_data) == 50000 and len(test_data) == 10000
        return train_data, test_data


class NB201CIFAR10Data(CIFAR10Data):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(self, root, train_transform, test_transform):
        # TODO: Complete
        train_data = ...
        test_data = ...

        return train_data, test_data


class NB201CIFAR100Data(CIFAR100Data):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(self, root, train_transform, test_transform):
        # TODO: Complete
        train_data = ...
        test_data = ...

        return train_data, test_data


class ImageNet16Data(ImageNetData):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(self, root, train_transform, test_transform):
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
        return train_data, test_data


class ImageNet16120Data(ImageNetData):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(self, root, train_transform, test_transform):
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000

        return train_data, test_data


class NB201ImageNet16120Data(ImageNet16120Data):

    def __init__(self, root, cutout, train_portion=1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(self, root, train_transform, test_transform):
        # TODO: Complete
        train_data = ...
        test_data = ...

        return train_data, test_data

class FASHIONMNISTData(ExperimentData):
    def __init__(self, root, train_portion=0.5):
        super().__init__(root, train_portion)

    def build_datasets(self):
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(
            self.root, train_transform, test_transform
        )

        if self.train_portion < 1:
            num_train = len(train_data)
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])
            return (train_data, train_sampler), (train_data, val_sampler), test_data
        else:
            return (train_data, None), (None, None), test_data

    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2859,), (0.3530,)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2859,), (0.3530,)),
        ])

        return train_transform, test_transform

    def load_datasets(self, root, train_transform, test_transform):
        train_data = dset.FashionMNIST(
            root, train=True, download=True,
            transform=train_transform
        )
        test_data = dset.FashionMNIST(
            root, train=False, download=True,
            transform=test_transform
        )

        return train_data, test_data
