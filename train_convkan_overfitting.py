import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import shared logic
from common import (
    get_convkan_model, get_small_fixed_dataset,
    prepare_dataset_root, CONVKAN_OVERFITTING_PATH, ensure_parent_dir
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training ConvKAN on limited data (overfitting experiment) on: {device}\n")

LEARNING_RATE = 1e-3
SAMPLES_PER_CLASS = 250
OVERFIT_BATCH_SIZE = 16  # Smaller batch size for limited dataset

# 2. LOAD & CLEAN DATASET
real_root = prepare_dataset_root()

# 3. LOAD SMALL DATASET (250 samples per class = 500 total)
# Note: Training set has data augmentation applied (see common.py)
train_dataset, val_dataset = get_small_fixed_dataset(real_root, samples_per_class=SAMPLES_PER_CLASS)

train_loader = DataLoader(train_dataset, batch_size=OVERFIT_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=OVERFIT_BATCH_SIZE, shuffle=False)

print(f"Training on {len(train_dataset)} samples ({SAMPLES_PER_CLASS * 2} total, 80% = {int(SAMPLES_PER_CLASS * 2 * 0.8)} per class)")
print(f"Validating on {len(val_dataset)} samples (20% = {int(SAMPLES_PER_CLASS * 2 * 0.2)} per class)")
print(f"Using batch size: {OVERFIT_BATCH_SIZE}")
print(f"Data augmentation: ENABLED (training set only)\n")

# 4. INIT MODEL
model = get_convkan_model(device, version="simple")
# Use class weights to ensure balanced learning
class_weights = torch.tensor([1.0, 1.0], device=device)  # Equal weights for both classes
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 5. TRAINING LOOP WITH OVERFITTING DETECTION
print("\nStarting training until overfitting is detected...\n")

train_losses = []
val_losses = []
train_accs = []
val_accs = []

patience = 5  # Number of epochs with no improvement before stopping
best_val_loss = float('inf')
best_model_state = None  # Store best model state
best_epoch = 0
epochs_no_improve = 0
epoch = 0
max_epochs = 100  # Safety limit

while epochs_no_improve < patience and epoch < max_epochs:
    epoch += 1
    
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(y_hat, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            val_loss += loss.item()
            _, predicted = torch.max(y_hat, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Check for overfitting (validation loss increasing)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()  # Save best model
        best_epoch = epoch
        epochs_no_improve = 0
        print(f"           ✓ Validation loss improved! Saved checkpoint.")
    else:
        epochs_no_improve += 1
        print(f"           ⚠ Validation loss NOT improved ({epochs_no_improve}/{patience})")

print(f"\n{'='*80}")
print(f"Training stopped after {epoch} epochs")
if epochs_no_improve >= patience:
    print(f"Reason: Validation loss did not improve for {patience} consecutive epochs (overfitting detected)")
else:
    print(f"Reason: Reached maximum epochs ({max_epochs})")
print(f"{'='*80}\n")

# 6. SAVE BEST MODEL
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model from epoch {best_epoch} (val loss: {best_val_loss:.4f})")
ensure_parent_dir(CONVKAN_OVERFITTING_PATH)
torch.save(model.state_dict(), CONVKAN_OVERFITTING_PATH)
print(f"Best model saved to {CONVKAN_OVERFITTING_PATH}")

# 7. PLOT TRAINING CURVES
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(range(1, epoch + 1), train_losses, marker='o', label='Training Loss', linewidth=2)
axes[0].plot(range(1, epoch + 1), val_losses, marker='s', label='Validation Loss', linewidth=2)
axes[0].axvline(x=epoch - epochs_no_improve, color='red', linestyle='--', label='Overfitting Point', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('ConvKAN: Training vs Validation Loss\n(Limited Data: 250 samples/class)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(range(1, epoch + 1), train_accs, marker='o', label='Training Accuracy', linewidth=2)
axes[1].plot(range(1, epoch + 1), val_accs, marker='s', label='Validation Accuracy', linewidth=2)
axes[1].axvline(x=epoch - epochs_no_improve, color='red', linestyle='--', label='Overfitting Point', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('ConvKAN: Training vs Validation Accuracy\n(Limited Data: 250 samples/class)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig('training_curves_convkan_overfitting.png', dpi=150, bbox_inches='tight')
print("Training curves saved to training_curves_convkan_overfitting.png")
plt.show()

print("\nConvKAN overfitting experiment complete!")
