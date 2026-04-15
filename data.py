import os
import sys

import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, transforms

import kagglehub
from dotenv import load_dotenv

from config import IMG_SIZE, MALARIA_DATASETS


# ---------------------------------------------------------------------------
# Dataset wrapper

class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a per-sample transform at load time.

    Walks the chain of nested Subsets to reach the underlying ImageFolder
    so that images are loaded fresh from disk (rather than from a cached,
    already-transformed sample).
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        actual_idx = self.subset.indices[idx] if hasattr(self.subset, "indices") else idx
        current = self.subset.dataset
        while hasattr(current, "dataset"):
            if hasattr(current, "indices"):
                actual_idx = current.indices[actual_idx]
            current = current.dataset
        img_path, label = current.samples[actual_idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transform helpers

def _base_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _augment_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Data splitting helpers

def find_data_root(start_path: str) -> str:
    """Walk *start_path* until a directory containing 'Uninfected' is found."""
    for root, dirs, _ in os.walk(start_path):
        if "Uninfected" in dirs:
            return root
    return start_path


def get_data_split(root_path: str, percentage: float = 1.0):
    """Return (train_dataset, val_dataset) from an ImageFolder root.

    10 % of total data is held out as a fixed test set; the remainder is
    shuffled and split 80/20 into train/validation.  *percentage* can be used
    to sub-sample the trainable portion for experiments.
    """
    full_dataset = datasets.ImageFolder(root=root_path, transform=_base_transform())
    print(f"Detected classes: {full_dataset.class_to_idx}")
    torch.manual_seed(42)

    total_size = len(full_dataset)
    test_size = int(0.1 * total_size)
    trainable_size = total_size - test_size
    trainable_set, _ = random_split(full_dataset, [trainable_size, test_size])

    exp_size = int(percentage * trainable_size)
    if exp_size < trainable_size:
        exp_set, _ = random_split(trainable_set, [exp_size, trainable_size - exp_size])
    else:
        exp_set = trainable_set

    train_size = int(0.8 * len(exp_set))
    return random_split(exp_set, [train_size, len(exp_set) - train_size])


def get_test_set(root_path: str):
    """Return the fixed 10 % hold-out test set for a given ImageFolder root."""
    full_dataset = datasets.ImageFolder(root=root_path, transform=_base_transform())
    torch.manual_seed(42)
    test_size = int(0.1 * len(full_dataset))
    _, test_set = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    return test_set


def get_small_fixed_dataset(root_path: str, samples_per_class: int = 20):
    """Return (train, val) subsets capped at *samples_per_class* per class.

    Training split uses augmentation; validation uses the base transform.
    """
    full_dataset = datasets.ImageFolder(root=root_path)
    torch.manual_seed(42)

    test_size = int(0.1 * len(full_dataset))
    trainable_set, _ = random_split(full_dataset, [len(full_dataset) - test_size, test_size])

    indices: dict[int, list[int]] = {0: [], 1: []}
    for i in range(len(trainable_set)):
        _, label = trainable_set[i]
        if len(indices[label]) < samples_per_class:
            indices[label].append(i)
        if all(len(v) == samples_per_class for v in indices.values()):
            break

    subset = torch.utils.data.Subset(trainable_set, indices[0] + indices[1])
    train_size = int(0.8 * len(subset))
    train_part, val_part = random_split(subset, [train_size, len(subset) - train_size])

    return (
        AugmentedDataset(train_part, _augment_transform()),
        AugmentedDataset(val_part, _base_transform()),
    )


# ---------------------------------------------------------------------------
# Dataset download / path resolution

def get_dataset() -> str:
    """Return a local path to the malaria dataset.

    Accepts an optional CLI argument: either a local path or a 1-based index
    into MALARIA_DATASETS.  Falls back to downloading the first entry.
    """
    # Download Dataset
    url = MALARIA_DATASETS[0]
    if len(sys.argv) == 2:
        argument = sys.argv[1]
        if os.path.exists(argument):
            return argument
        try:
            idx = int(argument)
            if 1 <= idx <= len(MALARIA_DATASETS):
                url = MALARIA_DATASETS[idx - 1]
        except ValueError:
            pass

    load_dotenv()
    print(f"Downloading {url}")
    data_path = kagglehub.dataset_download(url)
    print(f"Dataset path is located at {data_path}")
    return data_path


def prepare_dataset_root(dataset_path: str | None = None) -> str:
    """Resolve and return the directory that directly contains class sub-folders."""
    if dataset_path is None:
        dataset_path = get_dataset()
    real_root = find_data_root(dataset_path)
    print(f"Parent image path {real_root}")
    return real_root