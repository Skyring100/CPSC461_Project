import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import shared logic
from common import (
    get_convkan_model, get_data_split,
    prepare_dataset_root, CONVKAN_EXPERIMENT_TEMPLATE, format_experiment_path, BATCH_SIZE
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training ConvKAN experiments on: {device}")

LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

# Data percentages: 10%, 20%, ..., 90%
PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 2. LOAD & CLEAN DATASET
real_root = prepare_dataset_root()

# 3. RUN EXPERIMENTS
for experiment_num, percentage in enumerate(PERCENTAGES, 1):
    print(f"\n{'='*60}")
    print(f"Experiment {experiment_num}/9: Training on {int(percentage*100)}% of data")
    print(f"{'='*60}")
    
    # Load data for this experiment
    train_dataset, val_dataset = get_data_split(real_root, percentage=percentage)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples (80% of {int(percentage*100)}%): {len(train_dataset)}")
    print(f"Validation samples (20% of {int(percentage*100)}%): {len(val_dataset)}")
    
    # Init model
    model = get_convkan_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
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
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Results | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
    
    # Save model
    save_path = format_experiment_path(CONVKAN_EXPERIMENT_TEMPLATE, percentage)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

print(f"\n{'='*60}")
print("All ConvKAN experiments completed!")
print(f"{'='*60}")
