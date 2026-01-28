import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import shared logic
from common import (
    get_cnn_model, find_data_root, get_data_split, 
    get_dataset, BATCH_SIZE
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training CNN Baseline on: {device}")

CNN_SAVE_PATH = "malaria_cnn.pth"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

# 2. LOAD DATA (Exact same split as ConvKAN)
real_root = find_data_root(get_dataset())
train_dataset, test_dataset = get_data_split(real_root)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. INIT CNN MODEL
model = get_cnn_model(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. TRAINING LOOP
print("\nStarting CNN Training...")
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

    # Validation
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
    print(f"Epoch {epoch+1} Results | Val Acc: {acc:.2f}%")

# 5. SAVE
torch.save(model.state_dict(), CNN_SAVE_PATH)
print(f"\nCNN Model saved to {CNN_SAVE_PATH}")