from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
import copy
from medmnist import INFO, Evaluator
from common import (MODELS_ROOT, get_model, get_data_split, ensure_parent_dir)
import os
import numpy as np

def train_model(model_name: str, version: str):
    isConvKAN = True
    if (model_name.lower() == "convkan"):
        isConvKAN = True
    elif (model_name.lower() == "cnn"):
        isConvKAN = False
    else:
        print("Invalid model name, exiting")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device, isConvKAN, version)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    MODEL_PATH = os.path.join(MODELS_ROOT, "", "malaria_{model_name}_{VERSION}.pth")
    DATASET_PATH = os.path.join(os.getcwd(), "medmnist", version)
    print(DATASET_PATH)
    os.makedirs(DATASET_PATH, exist_ok=True)

    MAX_EPOCHS = 100
    PATIENCE = 5
    BATCH_SIZE = 128

    info = INFO[version]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    print("Downloading dataset")
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root=DATASET_PATH)
    test_dataset = DataClass(split='test', transform=data_transform, download=True, root=DATASET_PATH)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    print(train_dataset)
    
    print(f"Starting Training for {model_name} with version {version}")
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); 
            
            y_hat = model(x); 
            loss = criterion(y_hat, y)
            
            loss.backward(); optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x); _, predicted = torch.max(y_hat, 1)
                total += y.size(0); correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_PATH)



train_model("cnn", "nodulemnist3d")    