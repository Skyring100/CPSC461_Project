import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import shared logic
from common import (
    get_data_split,
    prepare_dataset_root, BATCH_SIZE,
    ensure_parent_dir
)

def train_model_experiments(model_name : str, model: nn.Sequential, device):
    base_path = os.path.join("models", "data_subsets", model_name.lower())
    # 1. SETUP

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
            print(f"Epoch {epoch+1} Results | Val Acc: {acc:.2f}%")
        
        # Save model
        save_path = os.path.join(base_path, f"malaria_{model_name}_subset_{percentage}.pth")
        ensure_parent_dir(save_path)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    print(f"\n{'='*60}")
    print(f"All {model_name} experiments completed!")
    print(f"{'='*60}")
