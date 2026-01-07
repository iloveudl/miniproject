from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision import transforms

from process_data import (Data as MNISTData, worker_init, resolve_dataset, RegressionMNIST, balanced_split)
from models import CNN


class DataRegression:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.numpy_generator = rng
        self.dataset_name = str(getattr(cfg, "regression_dataset", getattr(cfg, "dataset", "mnist"))).lower()
        self.train_batch_size = int(cfg.train_batch_size)
        self.test_batch_size = int(cfg.test_batch_size)
        self.gen = torch.Generator().manual_seed(int(cfg.seed))

        pool_ood_ratio = float(getattr(cfg, "pool_ood_ratio", 0.0))
        label_noise_eta = float(getattr(cfg, "label_noise", 0.0))

        mnist_like = {"mnist", "fashion_mnist", "fashion-mnist", "fashion"}
        if self.dataset_name in mnist_like:
            cfg.dataset = self.dataset_name if self.dataset_name != "fashion" else "fashion_mnist"
            self.init_mnist(cfg, rng)
        else:
            raise ValueError("Unknown regression dataset")

    def init_mnist(self, cfg, rng):
        ood_name = getattr(cfg, "ood_regression_dataset", "")
        rho = float(getattr(cfg, "pool_ood_ratio", 0.0))

        if not ood_name or rho <= 0.0:
            base = MNISTData(cfg, rng, regression=True)
            self._mnist = base
            self.train_indexes = base.train_indexes
            self.pool_indexes = base.pool_indexes
            self.test_loader = base.test_loader
            self.val_loader = base.val_loader
            self.output_dim = 10
            self.is_ood = torch.zeros(len(base.train_ds), dtype=torch.bool)
            self.pool_ood_ratio = 0.0
            self.make_model = lambda: CNN(conv_dropout=0.0, fc_dropout=0.0)
            return

        id_cls = resolve_dataset(cfg.dataset)
        ood_cls = resolve_dataset(ood_name)

        tfm = transforms.ToTensor()
        id_train = id_cls(cfg.data, train=True, download=True, transform=tfm)
        ood_train = ood_cls(cfg.data, train=True, download=True, transform=tfm)
        id_test = id_cls(cfg.data, train=False, download=True, transform=tfm)

        id_train_ds = RegressionMNIST(id_train)
        ood_train_ds = RegressionMNIST(ood_train)
        id_test_ds = RegressionMNIST(id_test)

        train_idx_id, rest_id = balanced_split(id_train.targets, 10, cfg.initial_class_size, rng)
        val_idx_id = rest_id[:cfg.val_size]
        pool_idx_id = rest_id[cfg.val_size:]

        id_len = len(id_train_ds)
        ood_len = len(ood_train_ds)
        ood_offset = id_len

        if rho >= 1.0:
            ood_pool_size = min(ood_len, len(pool_idx_id))
        else:
            ood_pool_size = int(round(len(pool_idx_id) * rho / max(1e-8, (1 - rho))))
            ood_pool_size = min(ood_pool_size, ood_len)

        ood_all_idx = torch.randperm(ood_len, generator=self.gen).tolist()
        pool_idx_ood = [(i + ood_offset) for i in ood_all_idx[:ood_pool_size]]

        self.train_ds = ConcatDataset([id_train_ds, ood_train_ds])
        self.test_loader = DataLoader(
            id_test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=self.gen,
        )

        self.val_loader = None
        if len(val_idx_id):
            self.val_loader = DataLoader(
                Subset(id_train_ds, val_idx_id),
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False,
                worker_init_fn=worker_init,
                generator=self.gen,
            )

        self.train_indexes = train_idx_id
        self.pool_indexes = pool_idx_id + pool_idx_ood
        self.is_ood = torch.zeros(id_len + ood_len, dtype=torch.bool)
        self.is_ood[id_len:] = True
        total_pool = len(self.pool_indexes)
        self.pool_ood_ratio = len(pool_idx_ood) / total_pool if total_pool > 0 else 0.0

        self.output_dim = 10
        self.make_model = lambda: CNN(conv_dropout=0.0, fc_dropout=0.0)

    def apply_label_noise(self, indices):
        mnist_like = {"mnist", "fashion_mnist", "fashion-mnist", "fashion"}
        if self.dataset_name in mnist_like and hasattr(self, "_mnist"):
            self._mnist.apply_label_noise(indices)

    def build_loaders(self, train_idx, pool_idx):
        mnist_like = {"mnist", "fashion_mnist", "fashion-mnist", "fashion"}
        if self.dataset_name in mnist_like:
            if hasattr(self, "_mnist"):
                return self._mnist.build_loaders(train_idx, pool_idx)
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
        raise ValueError(f"Unknown dataset: {self.dataset_name}")
