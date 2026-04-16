import os
import torch
import torchio as tio
import kagglehub
import medmnist
import shutil

from medmnist import INFO
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from config import IMG_SIZE, MALARIA_DATASETS



class AugmentedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)



# ---------------------------------------------------------------------------
# 2D Transforms (Malaria)

def get_2d_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_2d_val_transforms():
    """Basic transforms for validation: Resize and Normalize only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_malaria_loaders(batch_size):
    root = prepare_malaria_root()
    full_dataset = datasets.ImageFolder(root=root)
    
    # 72% Train, 18% Val, 10% Test
    total_len = len(full_dataset)
    train_size = int(0.72 * total_len)
    val_size = int(0.18 * total_len)
    test_size = total_len - train_size - val_size
    
    torch.manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_ds = AugmentedDataset(train_subset, get_2d_transforms())
    val_ds = AugmentedDataset(val_subset, get_2d_val_transforms())
    test_ds = AugmentedDataset(test_subset, get_2d_val_transforms())
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )



# ---------------------------------------------------------------------------
# 3D Transforms (Medmnist)

def get_3d_transforms():
    return transforms.Compose([
        # Convert to FloatTensor (C, D, H, W)
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RandomNoise(std=(0, 0.05)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_3d_val_transforms():
    return transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_medmnist_loaders(batch_size):
    DataClass = getattr(medmnist, INFO["nodulemnist3d"]["python_class"])
    
    root = prepare_medmnist_root()
    
    train_ds = DataClass(split="train", download=True, transform=get_3d_transforms(), root=root)
    val_ds = DataClass(split="val", download=True, transform=get_3d_val_transforms(), root=root)
    test_ds = DataClass(split="test", download=True, transform=get_3d_val_transforms(), root=root)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False)



# ---------------------------------------------------------------------------
# Prepare datasets

def prepare_malaria_root() -> str:
    data_path = kagglehub.dataset_download(MALARIA_DATASETS[0])

    # Find the root directory containing the dataset
    target_root = None
    for root, dirs, _ in os.walk(data_path):
        if "Uninfected" in dirs and "Parasitized" in dirs:
            target_root = root
            break

    # Remove redundant cell_images folder if it exists
    if target_root:
        redundant_path = os.path.join(target_root, "cell_images")
        if os.path.exists(redundant_path) and os.path.isdir(redundant_path):
            shutil.rmtree(redundant_path)
        return target_root
        
    return data_path


def prepare_medmnist_root() -> str:
    root = os.path.join(os.getcwd(), "medmnist")
    os.makedirs(root, exist_ok=True)
    return root
