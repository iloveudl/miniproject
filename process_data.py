import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


def resolve_dataset(name):
    name = name.strip().lower()
    if name in ("mnist",):
        return datasets.MNIST
    if name in ("fashion_mnist", "fashion-mnist", "fashion"):
        return datasets.FashionMNIST
    raise ValueError("Unknown dataset")


class RegressionMNIST(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, lbl = self.base[i]
        y = torch.zeros(10)
        y[lbl] = 1.0
        return img, y


class NoisyLabelDataset(Dataset):

    def __init__(self, base, base_labels, noisy_labels, num_classes=10, regression=False):
        self.base = base
        self.base_labels = np.asarray(base_labels)
        self.noisy_labels = noisy_labels
        self.num_classes = int(num_classes)
        self.regression = regression

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        true_cls = int(self.base_labels[idx])
        cls = int(self.noisy_labels.get(idx, true_cls))
        if self.regression:
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            y[cls] = 1.0
        else:
            y = cls
        return img, y


def balanced_split(labels, n_classes, n_per_class, rng):
    labels = np.asarray(labels)
    train, rest = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        train.append(idx[:n_per_class])
        rest.append(idx[n_per_class:])
    train = np.concatenate(train)
    rest = np.concatenate(rest)
    rng.shuffle(train)
    rng.shuffle(rest)
    return train.tolist(), rest.tolist()


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class Data:
    def __init__(self, cfg, rng, regression=False):
        self.numpy_generator = rng
        self.train_batch_size = int(cfg.train_batch_size)
        self.test_batch_size = int(cfg.test_batch_size)
        self.gen = torch.Generator().manual_seed(int(cfg.seed))

        self.label_noise_eta = float(getattr(cfg, "label_noise", 0.0))
        self.num_classes = int(getattr(cfg, "num_classes", 10))
        self.noisy_labels = {}

        dataset_name = str(getattr(cfg, "dataset", "mnist"))
        dataset_cls = resolve_dataset(dataset_name)

        base_train = dataset_cls(cfg.data, train=True, download=True, transform=transforms.ToTensor())
        base_test = dataset_cls(cfg.data, train=False, download=True, transform=transforms.ToTensor())
        self.base_labels = np.asarray(base_train.targets)

        self.train_ds = NoisyLabelDataset(
            base_train, self.base_labels, self.noisy_labels, num_classes=self.num_classes, regression=regression
        )
        self.test_ds = RegressionMNIST(base_test) if regression else base_test

        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=self.gen,
        )

        self.train_indexes, rest = balanced_split(base_train.targets, 10, cfg.initial_class_size, rng)
        val_idx = rest[:cfg.val_size]
        self.pool_indexes = rest[cfg.val_size:]

        if self.label_noise_eta > 0.0:
            self.apply_label_noise(self.train_indexes)

        self.val_loader = None
        if val_idx:
            self.val_loader = DataLoader(
                Subset(self.train_ds, val_idx),
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False,
                worker_init_fn=worker_init,
                generator=self.gen,
            )

    def draw_noisy_label(self, true_cls):
        if self.label_noise_eta <= 0.0:
            return true_cls
        if self.numpy_generator.random() >= self.label_noise_eta:
            return true_cls
        choices = [c for c in range(self.num_classes) if c != true_cls]
        return int(self.numpy_generator.choice(choices))

    def apply_label_noise(self, indices):
        if self.label_noise_eta <= 0.0:
            return
        for idx in indices:
            true_lbl = int(self.base_labels[idx])
            self.noisy_labels[idx] = self.draw_noisy_label(true_lbl)

    def build_loaders(self, train_idx, pool_idx):
        train = DataLoader(
            Subset(self.train_ds, train_idx),
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=self.gen,
        )
        pool = DataLoader(
            Subset(self.train_ds, pool_idx),
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=self.gen,
        )
        return train, pool
