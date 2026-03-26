import os
import shutil
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split
from convkan import ConvKAN, LayerNorm2D
from ConvKAN3D.ConvKAN3D import effConvKAN3D
import kagglehub
from dotenv import load_dotenv
from PIL import Image

# --- CONFIGURATION ---
IMG_SIZE = 48 
BATCH_SIZE = 16 
MODELS_ROOT = "models"

MALARIA_DATASETS = ["iarunava/cell-images-for-detecting-malaria", "nipunarora8/malaria-detection-dataset"]

LOADED_DATASET_INDEX = "local"

# --- HELPER CLASSES ---

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        actual_idx = self.subset.indices[idx] if hasattr(self.subset, 'indices') else idx
        current = self.subset.dataset
        while hasattr(current, 'dataset'):
            if hasattr(current, 'indices'): actual_idx = current.indices[actual_idx]
            current = current.dataset
        img_path, label = current.samples[actual_idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

# --- PATH HELPERS ---

def ensure_parent_dir(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def get_overfit_sweep_path(model_type, samples_per_class):
    root = os.path.join(MODELS_ROOT, "overfit_trials", model_type.lower())
    return os.path.join(root, f"malaria_{model_type.lower()}_overfit_{samples_per_class}samples.pth")

CNN_SAVE_PATH = os.path.join(MODELS_ROOT, "baseline", "cnn", "malaria_cnn.pth")
CONVKAN_SAVE_PATH = os.path.join(MODELS_ROOT, "baseline", "convkan", "malaria_convkan.pth")

# --- MODEL ARCHITECTURE ---

def add_block(model, isConvKAN, in_ch, out_ch, version="standard"):
    if isConvKAN:
        # Nano uses linear splines (order 1) on a 2-point grid for absolute minimum weight
        if version == "nano":
            g_size, s_order = 2, 1 
        elif version == "pico":
            g_size, s_order = 2, 2
        elif version == "android":
            g_size, s_order = 3, 2
        else:
            g_size, s_order = 5, 3
            
        model.append(ConvKAN(in_ch, out_ch, kernel_size=3, stride=1, padding=1, 
                             grid_size=g_size, spline_order=s_order))
        model.append(LayerNorm2D(out_ch))
    else:
        model.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        model.append(nn.BatchNorm2d(out_ch))
        model.append(nn.LeakyReLU())

def add3d_block(model, isConvKAN, in_ch, out_ch, version="standard"):
    if isConvKAN:
        # Nano uses linear splines (order 1) on a 2-point grid for absolute minimum weight
        if version == "nano":
            g_size, s_order = 2, 1 
        elif version == "pico":
            g_size, s_order = 2, 2
        elif version == "android":
            g_size, s_order = 3, 2
        else:
            g_size, s_order = 5, 3
            
        model.append(effConvKAN3D(in_ch, out_ch, kernel_size=3, stride=1, padding=1, 
                             grid_size=g_size, spline_order=s_order))
    else:
        model.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        model.append(nn.LeakyReLU())

def get_model(device, isConvKAN: bool, version="standard"):
    model = nn.Sequential()
    
    is3d_dataset = False
    # --- PARAMETER BUDGETING (TRUE PARITY TUNING) ---
    if version == "nano":
        if isConvKAN:
            # Atomic KAN (~2,154 parameters)
            channels = [3, 2, 4, 4, 4] 
        else:
            # Lean CNN (~2,256 parameters) - Tuned for parity
            channels = [3, 4, 8, 10, 12]
    elif version == "pico":
        channels = [3, 4, 8, 12, 16]
    elif version == "android":
        channels = [3, 8, 16, 16, 32]
    elif version == "simple":
        channels = [3, 16, 32, 32, 64]
    elif version == "nodulemnist3d":
        is3d_dataset = True
        if isConvKAN:
            channels = [1, 2, 4, 4, 4] 
        else:
            channels = [1, 4, 8, 10, 12]
    else:
        channels = [3, 32, 64, 128, 256]
        
    current_in = channels[0]
    if not is3d_dataset:
        for out_ch in channels[1:]:
            add_block(model, isConvKAN, current_in, out_ch, version=version)
            model.append(nn.MaxPool2d(2))
            current_in = out_ch

        # Decision Head Logic
        if version in ["android", "pico", "nano"]:
            if isConvKAN:
                g = 2 if version in ["pico", "nano"] else 3
                model.append(ConvKAN(channels[-1], 2, kernel_size=1, grid_size=g))
                model.append(nn.AdaptiveAvgPool2d(1))
                model.append(nn.Flatten())
            else:
                model.append(nn.AdaptiveAvgPool2d(1))
                model.append(nn.Flatten())
                model.append(nn.Linear(channels[-1], 2))
        else:
            model.append(nn.Flatten())
            with torch.no_grad():
                dummy_x = torch.zeros(1, channels[0], IMG_SIZE, IMG_SIZE)
                backbone_output = model[:-1](dummy_x) 
                flatten_size = backbone_output.numel()
            
            if isConvKAN:
                model.append(nn.Unflatten(1, (channels[-1], 1, 1)))
                model.append(ConvKAN(channels[-1], 2, kernel_size=1))
                model.append(nn.AdaptiveAvgPool2d(1))
                model.append(nn.Flatten())
            else:
                if version == "simple":
                    model.append(nn.Linear(flatten_size, 2))
                else:
                    model.append(nn.Linear(flatten_size, 256))
                    model.append(nn.LeakyReLU())
                    model.append(nn.Linear(256, 2))
    else:
        for out_ch in channels[1:]:
            add3d_block(model, isConvKAN, current_in, out_ch, version=version)
            current_in = out_ch
        
        # Final output layer
        if isConvKAN:
            model.append(effConvKAN3D(channels[-1], 2, kernel_size=1, grid_size=2))
            model.append(nn.AdaptiveAvgPool3d(1))
            model.append(nn.Flatten())
        else:
            model.append(nn.AdaptiveAvgPool3d(1))
            model.append(nn.Flatten())
            model.append(nn.Linear(channels[-1], 2))
        


    return model.to(device)

def get_convkan_model(device, version="standard"):
    return get_model(device, True, version)

def get_cnn_model(device, version="standard"):
    return get_model(device, False, version)

# --- DATA HELPERS ---

def find_data_root(start_path):
    for root, dirs, _ in os.walk(start_path):
        if "Uninfected" in dirs: return root
    return start_path

def get_data_split(root_path, percentage=1.0):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(root=root_path, transform=transform)
    torch.manual_seed(42)
    total_size = len(full_dataset)
    final_test_size = int(0.1 * total_size)
    trainable_size = total_size - final_test_size
    trainable_set, _ = random_split(full_dataset, [trainable_size, final_test_size])
    
    exp_size = int(percentage * trainable_size)
    if exp_size < trainable_size:
        exp_set, _ = random_split(trainable_set, [exp_size, trainable_size - exp_size])
    else:
        exp_set = trainable_set
    
    train_size = int(0.8 * len(exp_set))
    return random_split(exp_set, [train_size, len(exp_set) - train_size])

def get_test_set(root_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(root=root_path, transform=transform)
    torch.manual_seed(42)
    test_size = int(0.1 * len(full_dataset))
    _, test_set = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    return test_set

def get_small_fixed_dataset(root_path, samples_per_class=20):
    basic_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    aug_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(root=root_path)
    torch.manual_seed(42)
    test_size = int(0.1 * len(full_dataset))
    trainable_set, _ = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    indices = {0: [], 1: []}
    for i in range(len(trainable_set)):
        _, label = trainable_set[i]
        if len(indices[label]) < samples_per_class:
            indices[label].append(i)
        if all(len(v) == samples_per_class for v in indices.values()): break
    subset = torch.utils.data.Subset(trainable_set, indices[0] + indices[1])
    train_size = int(0.8 * len(subset))
    train_p, val_p = random_split(subset, [train_size, len(subset) - train_size])
    return AugmentedDataset(train_p, aug_transform), AugmentedDataset(val_p, basic_transform)

def get_dataset():
    # Check if user supplied an argument
    url = MALARIA_DATASETS[0]
    if len(sys.argv) == 2: 
        argument = sys.argv[1]
        # If user supplied a dataset path, then use that path
        if os.path.exists(argument): 
            return argument
        # If user supplied a dataset number, then use the corresponding link
        elif int(argument) <= len(MALARIA_DATASETS):
            url = MALARIA_DATASETS[int(argument)-1]
            LOADED_DATASET_INDEX = MALARIA_DATASETS[int(argument)-1]
    load_dotenv()
    print("Downloading "+url)
    data_path = kagglehub.dataset_download(url)
    print("Dataset path is located at "+data_path)
    return data_path

def prepare_dataset_root(dataset_path=None):
    if dataset_path is None: dataset_path = get_dataset()
    real_root = find_data_root(dataset_path)
    print(f"Parent image path {real_root}")
    return real_root

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)