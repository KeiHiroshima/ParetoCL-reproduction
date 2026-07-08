import os
import urllib.request
import zipfile
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


def _prepare_tinyimagenet(root: str = "./data") -> str:
    """Download and reorganize TinyImageNet if not already present.

    Returns the path to the extracted tiny-imagenet-200 directory.
    """
    data_path = os.path.join(root, "tiny-imagenet-200")
    val_reorganized_flag = os.path.join(data_path, ".val_reorganized")

    if not os.path.exists(data_path):
        zip_path = os.path.join(root, "tiny-imagenet-200.zip")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        print(f"Downloading TinyImageNet from {url} ...")
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting TinyImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        os.remove(zip_path)

    # Reorganize val folder to match ImageFolder convention once
    if not os.path.exists(val_reorganized_flag):
        val_dir = os.path.join(data_path, "val")
        val_images_dir = os.path.join(val_dir, "images")
        val_annotations = os.path.join(val_dir, "val_annotations.txt")

        if os.path.exists(val_annotations) and os.path.exists(val_images_dir):
            print("Reorganizing TinyImageNet val split...")
            with open(val_annotations) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split("\t")
                img_name, class_name = parts[0], parts[1]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_images_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
            # Clean up empty images dir
            try:
                os.rmdir(val_images_dir)
            except OSError:
                pass

        # Write flag so we don't redo this
        with open(val_reorganized_flag, "w") as f:
            f.write("done")

    return data_path


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
        if self.dataset_name == "cifar100":
            CIFAR100(root="./data", train=True, download=True)
            CIFAR100(root="./data", train=False, download=True)
        elif self.dataset_name == "cifar10":
            CIFAR10(root="./data", train=True, download=True)
            CIFAR10(root="./data", train=False, download=True)
        elif self.dataset_name == "tinyimagenet":
            _prepare_tinyimagenet("./data")

    def setup(self, stage: Optional[str] = None):
        if self.dataset_name == "cifar100":
            full_train = CIFAR100(root="./data", train=True, transform=self.transform)
            full_test = CIFAR100(root="./data", train=False, transform=self.transform)
            targets_train = torch.tensor(full_train.targets)
            targets_test = torch.tensor(full_test.targets)
            self._build_task_splits_from_targets(
                full_train, full_test, targets_train, targets_test
            )

        elif self.dataset_name == "cifar10":
            full_train = CIFAR10(root="./data", train=True, transform=self.transform)
            full_test = CIFAR10(root="./data", train=False, transform=self.transform)
            targets_train = torch.tensor(full_train.targets)
            targets_test = torch.tensor(full_test.targets)
            self._build_task_splits_from_targets(
                full_train, full_test, targets_train, targets_test
            )

        elif self.dataset_name == "tinyimagenet":
            data_path = _prepare_tinyimagenet("./data")
            # ImageFolder sorts classes alphabetically; the class index == label
            full_train = ImageFolder(
                os.path.join(data_path, "train"), transform=self.transform
            )
            full_test = ImageFolder(
                os.path.join(data_path, "val"), transform=self.transform
            )
            targets_train = torch.tensor([s[1] for s in full_train.samples])
            targets_test = torch.tensor([s[1] for s in full_test.samples])
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
