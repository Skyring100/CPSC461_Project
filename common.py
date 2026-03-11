import os
import shutil
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split
from convkan import ConvKAN, LayerNorm2D
import kagglehub
from dotenv import load_dotenv

# Custom dataset wrapper for on-the-fly augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper that applies transform on-the-fly to a subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # Get the base dataset to access raw images
        base_dataset = subset
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        # Get the actual index and label from the subset
        actual_idx = self.subset.indices[idx] if hasattr(self.subset, 'indices') else idx
        
        # Navigate through nested subsets to get to the actual data
        current = self.subset.dataset
        current_idx = actual_idx
        while hasattr(current, 'dataset'):
            if hasattr(current, 'indices'):
                current_idx = current.indices[current_idx]
            current = current.dataset
        
        # Get raw image path and label from ImageFolder
        img_path, label = current.samples[current_idx]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, label

# --- CONFIGURATION ---
IMG_SIZE = 100
BATCH_SIZE = 32
CONVKAN_SAVE_PATH = "malaria_convkan.pth" 
CNN_SAVE_PATH = "malaria_cnn.pth"

# Experiment path templates (use format_experiment_path() to create)
CONVKAN_EXPERIMENT_TEMPLATE = "malaria_convkan_{pct}pct.pth"
CNN_EXPERIMENT_TEMPLATE = "malaria_cnn_{pct}pct.pth"

# Overfitting experiment paths
CONVKAN_OVERFITTING_PATH = "malaria_convkan_overfitting.pth"
CNN_OVERFITTING_PATH = "malaria_cnn_overfitting.pth"

def format_experiment_path(template, percentage):
    """Formats an experiment path with the given percentage (10, 20, ..., 90)."""
    return template.format(pct=int(percentage * 100))  
# --- MODEL ARCHITECTURE ---
# To calculate convolutional output layer height/width: (floor(last_size + 2x(Padding) - Kernel_size)/Stride)+1
# For simplicity, if padding=0 and stride=1, then we have: last_size - kernel_size + 1
# Using https://www.nature.com/articles/s41598-025-87979-5 for layer composition reference. They use CNNs for Malaria detection
def get_convkan_model(device, version=""):
    """
    Returns the compiled ConvKAN model.
    Using a function ensures main.py and evaluate.py match exactly.
    """
    return get_model(device, True, version)
    
    '''
    model = nn.Sequential(
        ConvKAN(3, 32, padding=1, kernel_size=3, stride=1),
        LayerNorm2D(32),
        nn.MaxPool2d(2), 
        
        ConvKAN(32, 64, padding=1, kernel_size=3, stride=1),
        LayerNorm2D(64),
        nn.MaxPool2d(2),
        
        ConvKAN(64, 128, padding=1, kernel_size=3, stride=1),
        LayerNorm2D(128),
        nn.MaxPool2d(2),

        ConvKAN(128, 256, padding=1, kernel_size=3, stride=1),
        LayerNorm2D(256),
        nn.MaxPool2d(2),       

        nn.Flatten(),
        # Linear is equivalent to dense layer
        nn.Linear(9216, 256),
        nn.LeakyReLU(),
        nn.Linear(256,2)
    ).to(device)

    return model
    '''

def get_cnn_model(device, version=""):
    """
    A Standard CNN Baseline.
    Structure mimics the ConvKAN
    """
    return get_model(device, False, version)

    '''
    model = nn.Sequential(
        nn.Conv2d(3, 32, padding=1, kernel_size=3, stride=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2), 
        
        nn.Conv2d(32, 64, padding=1, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, padding=1, kernel_size=3, stride=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, padding=1, kernel_size=3, stride=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),       

        nn.Flatten(),
        # Linear is equivalent to dense layer
        nn.Linear(9216, 256),
        nn.LeakyReLU(),
        nn.Linear(256,2)
    ).to(device)
    
    return model
    '''

def get_model(device, isConvKAN : bool, version=""):
    model = nn.Sequential()
    if version == "":
        add_convolutional_layer(model, isConvKAN, 3, 32, 3, 1, 1)
        if not isConvKAN:
            model.append(nn.LeakyReLU())
        model.append(nn.MaxPool2d(2)) 
        
        add_convolutional_layer(model, isConvKAN, 32, 64, 3, 1, 1)
        if not isConvKAN:
            model.append(nn.LeakyReLU())
        model.append(nn.MaxPool2d(2)) 
        
        add_convolutional_layer(model, isConvKAN, 64, 128, 3, 1, 1)
        if not isConvKAN:
            model.append(nn.LeakyReLU())
        model.append(nn.MaxPool2d(2)) 

        add_convolutional_layer(model, isConvKAN, 128, 256, 3, 1, 1)
        if not isConvKAN:
            model.append(nn.LeakyReLU())
        model.append(nn.MaxPool2d(2))     

        model.append(nn.Flatten())
        # Linear is equivalent to dense layer
        model.append(nn.Linear(9216, 256))
        model.appednd(nn.LeakyReLU())
        model.append(nn.Linear(256,2))
    elif version == "simple":
        add_convolutional_layer(model, isConvKAN, 3, 32, 3, 1, 1)
        if not isConvKAN:
            model.append(nn.LeakyReLU())
        model.append(nn.MaxPool2d(2)) 

        add_convolutional_layer(model, isConvKAN, 32, 64, 3, 1, 1)
        if not isConvKAN:
            model.append(nn.LeakyReLU())
        model.append(nn.MaxPool2d(2)) 

        model.append(nn.Flatten())
        model.append(nn.Linear(2304,2))

    return model.to(device)

def add_convolutional_layer(model : nn.Sequential, isConvKAN : bool, inChannel, outChannel, kernel_size, stride, padding):
    if isConvKAN:
        model.append(ConvKAN(inChannel, outChannel, kernel_size=kernel_size, stride=stride, padding=padding))
        model.append(LayerNorm2D(outChannel))
    else:
        model.append(nn.Conv2d(inChannel, outChannel, kernel_size=kernel_size, stride=stride, padding=padding))
        model.append(nn.BatchNorm2d(outChannel))

# --- DATA HELPERS ---
def find_data_root(start_path):
    """Recursively finds the folder containing the actual class subfolders."""
    for root, dirs, _ in os.walk(start_path):
        if "Parasitized" in dirs and "Uninfected" in dirs:
            return root
    return start_path

def get_data_split(root_path, percentage=1.0):
    """
    Loads data for experiments with variable training set sizes.
    - percentage: fraction of the training pool (90% of data) to use (0.1 to 1.0)
    - Returns: (train_set, val_set) with 80-20 split of the selected percentage
    
    CRITICAL: Uses torch.manual_seed(42) so the splits are identical every time.
    The first 10% is reserved for final testing and is NOT returned by this function.
    """
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(root=root_path, transform=transform)
    
    # LOCK THE SPLIT
    torch.manual_seed(42)
    
    # First, split off 10% for final testing (never returned by this function)
    total_size = len(full_dataset)
    final_test_size = int(0.1 * total_size)
    trainable_size = total_size - final_test_size
    
    # Split into trainable pool (90%) and held-out test set (10%)
    trainable_set, _ = random_split(full_dataset, [trainable_size, final_test_size])
    
    # Now split the trainable pool based on the requested percentage
    experiment_size = int(percentage * trainable_size)
    if experiment_size < trainable_size:
        experiment_set, _ = random_split(trainable_set, [experiment_size, trainable_size - experiment_size])
    else:
        experiment_set = trainable_set
    
    # Split experiment data into 80% train and 20% validation
    train_size = int(0.8 * len(experiment_set))
    val_size = len(experiment_set) - train_size
    
    train_set, val_set = random_split(experiment_set, [train_size, val_size])
    return train_set, val_set


def get_test_set(root_path):
    """
    Returns the held-out 10% test set for final evaluation.
    This set is NOT used during any training, only for final comparison.
    CRITICAL: Uses torch.manual_seed(42) so this matches the split in get_data_split().
    """
    
    # Data preprocessing (must match get_data_split)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(root=root_path, transform=transform)
    
    # LOCK THE SPLIT (must match get_data_split)
    torch.manual_seed(42)
    
    total_size = len(full_dataset)
    final_test_size = int(0.1 * total_size)
    trainable_size = total_size - final_test_size
    
    # Get the test set (last 10%)
    _, test_set = random_split(full_dataset, [trainable_size, final_test_size])
    return test_set


def get_small_fixed_dataset(root_path, samples_per_class=20):
    """
    Returns a small fixed dataset with a specific number of samples per class.
    Used for overfitting experiments to train on limited data.
    Applies data augmentation to training set to help models learn with limited data.
    
    Returns: (train_set, val_set) with 80-20 split of the small dataset
    CRITICAL: Uses torch.manual_seed(42) for reproducibility
    """
    
    # Data preprocessing WITHOUT augmentation (for validation)
    basic_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Data augmentation for training (applied on-the-fly)
    augmentation_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load full dataset WITHOUT transform (we'll apply it later)
    full_dataset = datasets.ImageFolder(root=root_path)
    
    # LOCK THE SPLIT
    torch.manual_seed(42)
    
    # Split off the 10% test set first (never use this data)
    total_size = len(full_dataset)
    final_test_size = int(0.1 * total_size)
    trainable_size = total_size - final_test_size
    trainable_set, _ = random_split(full_dataset, [trainable_size, final_test_size])
    
    # Now extract exactly samples_per_class from each class
    class_indices = {i: [] for i in range(len(full_dataset.classes))}
    for idx in range(len(trainable_set)):
        _, label = trainable_set[idx]
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
        # Stop early if we have enough
        if all(len(indices) >= samples_per_class for indices in class_indices.values()):
            break
    
    # Collect all selected indices
    selected_indices = []
    for class_id in sorted(class_indices.keys()):
        selected_indices.extend(class_indices[class_id][:samples_per_class])
    
    # Create subset
    small_dataset = torch.utils.data.Subset(trainable_set, selected_indices)
    
    # Split into 80% train and 20% validation
    train_size = int(0.8 * len(small_dataset))
    val_size = len(small_dataset) - train_size
    
    train_set, val_set = random_split(small_dataset, [train_size, val_size])
    
    # Wrap training set with augmentation
    train_set_augmented = AugmentedDataset(train_set, augmentation_transform)
    
    # Wrap validation set with basic transform (no augmentation)
    val_set_transformed = AugmentedDataset(val_set, basic_transform)
    
    return train_set_augmented, val_set_transformed

def get_dataset():
    """
    Downloads the Malaria image dataset automatically. Needs a Kaggle API key to work
    """

    # If path to dataset is provided in command line, use that instead
    if len(sys.argv) == 2:
        print(f"Using command line argument for dataset path: {sys.argv[1]}")
        return sys.argv[1]

    load_dotenv()
    kaggle_API_key = os.environ.get("KAGGLE_API_KEY")
    if not kaggle_API_key:
        print("A '.env' file needs to have a KAGGLE_API_KEY environment variable with a valid Kaggle API key")
        sys.exit()

    # If dataset already downloaded, kagglehub uses the one cached instead of downloading again
    print("Checking for dataset on system (will download if dataset not found)")
    path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
    print(f"Dataset retrieval successful, dataset found at {path}")
    return path

def clean_dataset_root(data_root):
    """
    Removes known nested garbage folders from the dataset tree.
    """
    garbage_folder = os.path.join(data_root, "cell_images")
    if os.path.isdir(garbage_folder):
        print(f"Removing garbage folder: {garbage_folder}")
        shutil.rmtree(garbage_folder)

def prepare_dataset_root(dataset_path=None):
    """
    Gets dataset path, resolves the real image root, and cleans known garbage folders.
    Returns the cleaned root directory to use for ImageFolder.
    """
    if dataset_path is None:
        dataset_path = get_dataset()

    real_root = find_data_root(dataset_path)
    clean_dataset_root(real_root)
    return real_root