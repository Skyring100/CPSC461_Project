import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split
from convkan import ConvKAN, LayerNorm2D
import kagglehub
from dotenv import load_dotenv

# --- CONFIGURATION ---
IMG_SIZE = 64
BATCH_SIZE = 32
MODEL_SAVE_PATH = "malaria_convkan.pth"

# --- MODEL ARCHITECTURE ---
def get_model(device):
    """
    Returns the compiled ConvKAN model.
    Using a function ensures main.py and evaluate.py match exactly.
    """
    model = nn.Sequential(
        # Layer 1: Input 3 (RGB) -> Output 32
        ConvKAN(3, 32, padding=1, kernel_size=3, stride=1),
        LayerNorm2D(32),
        nn.ReLU(),
        nn.MaxPool2d(2), # Reduces 64x64 -> 32x32
        
        # Layer 2: Input 32 -> Output 32
        ConvKAN(32, 32, padding=1, kernel_size=3, stride=1),
        LayerNorm2D(32),
        nn.ReLU(),
        nn.MaxPool2d(2), # Reduces 32x32 -> 16x16
        
        # Layer 3: Input 32 -> Output 2 (Classes)
        ConvKAN(32, 2, padding=1, kernel_size=3, stride=2),
        
        # Global Average Pooling -> Flatten
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    ).to(device)
    return model

def get_cnn_model(device):
    """
    A Standard CNN Baseline.
    Structure mimics the ConvKAN: 3 layers, roughly same depth.
    """
    model = nn.Sequential(
        # Layer 1: Input 3 (RGB) -> Output 32
        nn.Conv2d(3, 32, padding=1, kernel_size=3, stride=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2), # Reduces 64x64 -> 32x32
        
        # Layer 2: Input 32 -> Output 32
        nn.Conv2d(32, 32, padding=1, kernel_size=3, stride=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2), # Reduces 32x32 -> 16x16
        
        # Layer 3: Input 32 -> Output 2
        nn.Conv2d(32, 2, padding=1, kernel_size=3, stride=1),
        # We don't need MaxPool here if we use AdaptiveAvgPool right after
        
        # Global Average Pooling -> Flatten
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    ).to(device)
    return model

# --- DATA HELPERS ---
def find_data_root(start_path):
    """Recursively finds the folder containing the actual class subfolders."""
    for root, dirs, _ in os.walk(start_path):
        if "Parasitized" in dirs and "Uninfected" in dirs:
            return root
    return start_path

def get_data_split(root_path):
    """
    Loads data and returns (train_set, test_set).
    CRITICAL: Uses torch.manual_seed(42) so the split is identical every time.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    get_dataset()
    print(root_path)
    full_dataset = datasets.ImageFolder(root=root_path, transform=transform)
    
    # LOCK THE SPLIT
    torch.manual_seed(42)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_set, test_set = random_split(full_dataset, [train_size, test_size])
    return train_set, test_set

def get_dataset():
    """
    Downloads the Malaria image dataset automatically. Needs a Kaggle API key to work
    """
    load_dotenv()
    kaggle_API_key = os.environ.get("KAGGLE_API_KEY")
    if not kaggle_API_key:
        print("No Kaggle API key in '.env. file, which is needed to download the dataset.")
        SystemExit()

    # If dataset already downloaded, kagglehub uses the one cached instead of downloading again
    print("Checking for dataset on system (will download if dataset not found)")
    path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
    print("Dataset retrieval successful")
    return path