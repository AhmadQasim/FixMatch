import logging

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
matek_mean = (0.8205, 0.7279, 0.8360)
matek_std = (0.1719, 0.2589, 0.1042)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(root, test_root, num_labeled, num_expand_x, num_expand_u):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_expand_x, num_expand_u, num_classes=10, total_size=60000)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)
    logger.info("Dataset: CIFAR10")
    logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
                f" Unlabeled examples: {len(train_unlabeled_idxs)}"
                f" Test examples: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(root, test_root, num_labeled, num_expand_x, num_expand_u):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_classes=100, total_size=60000)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    logger.info("Dataset: CIFAR100")
    logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
                f" Unlabeled examples: {len(train_unlabeled_idxs)}"
                f" Test examples: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_matek(root, test_root, num_labeled, num_expand_x, num_expand_u):
    transform_labeled = transforms.Compose([
        transforms.Resize(size=64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=matek_mean, std=matek_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=matek_mean, std=matek_std)
    ])
    base_dataset = datasets.ImageFolder(
            root, transform=None
        )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_expand_x, num_expand_u, num_classes=15, total_size=len(base_dataset))

    train_labeled_dataset = MATEKSSL(root, train_labeled_idxs, transform=transform_labeled)

    train_unlabeled_dataset = MATEKSSL(root, train_unlabeled_idxs,
                                       transform=TransformFix(mean=matek_mean, std=matek_std))

    test_dataset = datasets.ImageFolder(test_root, transform=transform_val)
    logger.info("Dataset: MATEK")
    logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
                f" Unlabeled examples: {len(train_unlabeled_idxs)}"
                f" Test examples: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(labels,
              num_labeled,
              num_expand_x,
              num_expand_u,
              num_classes,
              total_size):
    unlabeled_ratio = 1-num_labeled/total_size
    labeled_idx, unlabeled_idx = train_test_split(
        np.arange(total_size),
        test_size=unlabeled_ratio,
        shuffle=True,
        stratify=labels)

    expand_labeled = num_expand_x // len(labeled_idx)
    expand_unlabeled = num_expand_u // len(unlabeled_idx)
    labeled_idx = np.hstack(
        [labeled_idx for _ in range(expand_labeled)])
    unlabeled_idx = np.hstack(
        [unlabeled_idx for _ in range(expand_unlabeled)])

    # doc: cover up the difference between the dataset size and the args.k_img for labeled and
    # args.k_img.args.mu for the unlabeled case
    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    if len(unlabeled_idx) < num_expand_u:
        diff = num_expand_u - len(unlabeled_idx)
        unlabeled_idx = np.hstack(
            (unlabeled_idx, np.random.choice(unlabeled_idx, diff)))
    else:
        assert len(unlabeled_idx) == num_expand_u

    return labeled_idx, unlabeled_idx


class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(size=64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
        ])
        self.strong = transforms.Compose([
            transforms.Resize(size=64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexes, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexes, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MATEKSSL(Dataset):
    def __init__(self, root, indexes, transform=None):
        self.transform = transform
        self.indexes = indexes
        self.dataset = datasets.ImageFolder(root, transform=self.transform)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img, target = self.dataset[self.indexes[index]]

        return img, target
