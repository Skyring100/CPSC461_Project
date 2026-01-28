import os
import shutil
import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import shared logic
from common import (
    get_model, find_data_root, get_data_split, 
    get_dataset, MODEL_SAVE_PATH, BATCH_SIZE
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

# 2. DOWNLOAD & CLEAN DATASET
if not os.path.exists(get_dataset()):
    print("Dataset not found. Downloading from Kaggle...")
    cached_path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
    shutil.move(cached_path, get_dataset())
    print(f"Dataset moved to: {get_dataset()}")

real_root = find_data_root(get_dataset())

# Fix: Remove the recursive garbage folder if it exists
garbage_folder = os.path.join(real_root, "cell_images")
if os.path.exists(garbage_folder):
    print(f"Removing garbage folder: {garbage_folder}")
    shutil.rmtree(garbage_folder)

# 3. LOAD DATA (SEED SECURED)
print(f"Loading data from: {real_root}")
train_dataset, test_dataset = get_data_split(real_root)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(test_dataset)}")

# 4. INIT MODEL
model = get_model(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 5. TRAINING LOOP
print("\nStarting Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Validation (using the test set as validation during training)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1} Results | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

# 6. SAVE
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")