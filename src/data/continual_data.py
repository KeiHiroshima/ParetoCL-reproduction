import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

# Hugging Face mirrors used instead of torchvision's default sources
# (www.cs.toronto.edu / cs231n.stanford.edu), which are single, non-CDN
# academic servers that are extremely slow/unreachable from some networks.
# HF Hub is CDN-backed and downloads at ~10-20x the speed in practice.
HF_DATASET_CONFIG = {
    "cifar10": dict(
        repo_id="uoft-cs/cifar10",
        image_key="img",
        label_key="label",
        test_split="test",
    ),
    "cifar100": dict(
        repo_id="uoft-cs/cifar100",
        image_key="img",
        label_key="fine_label",
        test_split="test",
    ),
    "tinyimagenet": dict(
        repo_id="zh-plus/tiny-imagenet",
        image_key="image",
        label_key="label",
        test_split="valid",
    ),
}

HF_CACHE_DIR = os.path.join("./data", "hf_datasets")


class _HFImageDataset(Dataset):
    """Adapts a Hugging Face `datasets.Dataset` split to the
    (transformed image tensor, integer label) interface torchvision
    datasets provide, so it plugs into `torch.utils.data.Subset`/`DataLoader`
    unchanged."""

    def __init__(self, hf_dataset, image_key: str, label_key: str, transform):
        self.hf_dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # `Subset.indices` are stored as a torch.Tensor (from `.nonzero()`), so
        # `idx` may arrive as a 0-d tensor; `datasets.Dataset.__getitem__`
        # only accepts plain Python ints/slices/lists.
        row = self.hf_dataset[int(idx)]
        image = row[self.image_key].convert("RGB")
        return self.transform(image), row[self.label_key]


class ReplayBuffer:
    def __init__(
        self, buffer_size: int, input_shape: Tuple[int, ...], num_classes: int
    ):
        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.current_size = 0

        # Pre-allocate memory
        self.images = torch.zeros((buffer_size, *input_shape), dtype=torch.float32)
        self.targets = torch.zeros(buffer_size, dtype=torch.long)
        self.task_ids = torch.zeros(buffer_size, dtype=torch.long)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.current_size == 0:
            return None, None, None

        indices = torch.randint(0, self.current_size, (batch_size,))
        return self.images[indices], self.targets[indices], self.task_ids[indices]


class ContinualDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "cifar100",
        batch_size: int = 32,
        num_tasks: int = 10,
        buffer_size: int = 32,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.buffer_size = buffer_size
        self.current_task_id = 0

        if dataset_name == "cifar100":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
            self.num_classes = 100
            self.input_shape = (3, 32, 32)
        elif dataset_name == "cifar10":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),
                ]
            )
            self.num_classes = 10
            self.input_shape = (3, 32, 32)
        elif dataset_name == "tinyimagenet":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
                    ),
                ]
            )
            self.num_classes = 200
            self.input_shape = (3, 64, 64)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.classes_per_task = self.num_classes // self.num_tasks
        self.buffer = ReplayBuffer(buffer_size, self.input_shape, self.num_classes)

        self.train_dataset = None
        self.test_dataset = None
        self.task_datasets = []  # List of (train_subset, test_subset) for each task

    def prepare_data(self):
        cfg = HF_DATASET_CONFIG[self.dataset_name]
        # Triggers (and caches) the download; cached on subsequent calls/runs.
        load_dataset(cfg["repo_id"], cache_dir=HF_CACHE_DIR)

    def setup(self, stage: Optional[str] = None):
        cfg = HF_DATASET_CONFIG[self.dataset_name]
        hf_splits = load_dataset(cfg["repo_id"], cache_dir=HF_CACHE_DIR)
        hf_train = hf_splits["train"]
        hf_test = hf_splits[cfg["test_split"]]

        full_train = _HFImageDataset(
            hf_train, cfg["image_key"], cfg["label_key"], self.transform
        )
        full_test = _HFImageDataset(
            hf_test, cfg["image_key"], cfg["label_key"], self.transform
        )
        # Column-only access (no image decoding) to get integer labels.
        targets_train = torch.tensor(hf_train[cfg["label_key"]])
        targets_test = torch.tensor(hf_test[cfg["label_key"]])

        self._build_task_splits_from_targets(
            full_train, full_test, targets_train, targets_test
        )

    def _build_task_splits_from_targets(
        self, full_train, full_test, targets_train, targets_test
    ):
        for t in range(self.num_tasks):
            start_class = t * self.classes_per_task
            end_class = (t + 1) * self.classes_per_task

            train_indices = (
                (targets_train >= start_class) & (targets_train < end_class)
            ).nonzero(as_tuple=True)[0]
            test_indices = (
                (targets_test >= start_class) & (targets_test < end_class)
            ).nonzero(as_tuple=True)[0]

            self.task_datasets.append(
                (Subset(full_train, train_indices), Subset(full_test, test_indices))
            )

    def set_task(self, task_id: int):
        self.current_task_id = task_id
        self.train_dataset = self.task_datasets[task_id][0]
        _ = self.task_datasets[task_id][1]  # test dataset for this task

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        # Return a list of dataloaders for all tasks seen so far
        dataloaders = []
        for i in range(self.current_task_id + 1):
            dataloaders.append(
                DataLoader(
                    self.task_datasets[i][1],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4,
                )
            )

        return dataloaders
