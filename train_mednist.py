from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import medmnist
import copy
from medmnist import INFO
from common import (MODELS_ROOT, get_model, get_data_split, ensure_parent_dir)
import os
import matplotlib.pyplot as plt
import sys

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
    print(model)
    
    MODEL_DIRECTORY = os.path.join(MODELS_ROOT, "mednist")
    if(not os.path.exists(MODEL_DIRECTORY)): 
        os.makedirs(MODEL_DIRECTORY)
    MODEL_PATH = os.path.join(MODEL_DIRECTORY, f'medmnist_{model_name}_{version}.pth')
    GRAPH_PATH = os.path.join(MODEL_DIRECTORY, f'medmnist_{model_name}_{version}_metrics.png')
    DATASET_PATH = os.path.join(os.getcwd(), "medmnist", version)
    print(f'Dataset path: {DATASET_PATH}')
    os.makedirs(DATASET_PATH, exist_ok=True)

    MAX_EPOCHS = 100
    PATIENCE = 5
    BATCH_SIZE = 20

    DataClass = getattr(medmnist, INFO[version]['python_class'])

    # load the data
    print("Downloading dataset")
    train_dataset = DataClass(split='train',  download=True, root=DATASET_PATH)
    test_dataset = DataClass(split='test',  download=True, root=DATASET_PATH)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Starting Training for {model_name} with version {version}")
    best_acc, epochs_no_improve = 0.0, 0
    history = {"train_loss": [], "val_acc": []}
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            # Since Mednist tensor was stored as a torch.double, we need to cast for out model
            # Mednist output label tensors are set to use mutliclasses, so we must squeeze them into 1D
            x, y = x.float().to(device), y.squeeze().long().to(device)
            optimizer.zero_grad();

            y_hat = model(x); 
            loss = criterion(y_hat, y)
            
            loss.backward(); optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.float().to(device), y.squeeze().long().to(device)
                y_hat = model(x); _, predicted = torch.max(y_hat, 1)
                total += y.size(0); correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total
        history["val_acc"].append(acc)
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

    #Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history["val_acc"], color='green', marker='o'); ax1.set_title("Validation Accuracy (%)")
    ax2.plot(history["train_loss"], color='orange'); ax2.set_title("Training Loss")
    plt.savefig(GRAPH_PATH); plt.close()


if len(sys.argv) == 1:
    train_model("cnn", "nodulemnist3d")    
    train_model("convkan", "nodulemnist3d")
elif len(sys.argv) == 2:
    train_model(sys.argv[1], "nodulemnist3d")
else:
    train_model(sys.argv[1], sys.argv[2])
