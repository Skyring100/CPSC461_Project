import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
import time
import psutil
import json

from common import (
    get_convkan_model, get_data_split, count_parameters,
    prepare_dataset_root, MODELS_ROOT, ensure_parent_dir
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = "nano" 
NEW_BATCH_SIZE = 16 
PATIENCE = 5
MAX_EPOCHS = 100

NANO_DIR = os.path.join(MODELS_ROOT, VERSION)
MODEL_PATH = os.path.join(NANO_DIR, f"malaria_convkan_{VERSION}.pth")
GRAPH_PATH = os.path.join(NANO_DIR, f"convkan_{VERSION}_metrics.png")
STATS_PATH = os.path.join(NANO_DIR, f"convkan_{VERSION}_stats.json")
ensure_parent_dir(MODEL_PATH)

# 2. DATA & MODEL
real_root = prepare_dataset_root()
train_dataset, test_dataset = get_data_split(real_root)
train_loader = DataLoader(train_dataset, batch_size=NEW_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NEW_BATCH_SIZE, shuffle=False)

model = get_convkan_model(device, version=VERSION)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. METRIC TRACKING START
start_time = time.perf_counter()
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()

# 4. TRAINING LOOP
history = {"train_loss": [], "val_acc": []}
best_acc, epochs_no_improve = 0.0, 0

print(f"\nStarting {VERSION.upper()} ConvKAN Training...")

for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); y_hat = model(x); loss = criterion(y_hat, y)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    history["train_loss"].append(epoch_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
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

# 5. FINAL METRICS & SAVE
end_time = time.perf_counter()
total_time = end_time - start_time
peak_mem = (torch.cuda.max_memory_allocated(device) if device.type == 'cuda' 
            else psutil.Process(os.getpid()).memory_info().rss) / (1024**2)

stats = {
    "model_type": "ConvKAN",
    "params": count_parameters(model),
    "total_time_sec": total_time,
    "avg_time_per_epoch": total_time / (epoch + 1),
    "peak_memory_mb": peak_mem,
    "best_val_acc": best_acc,
    "history": history
}

with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=4)

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_PATH)

# 6. DUAL PLOTTING
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history["val_acc"], color='green', marker='o'); ax1.set_title("Validation Accuracy (%)")
ax2.plot(history["train_loss"], color='orange'); ax2.set_title("Training Loss")
plt.savefig(GRAPH_PATH); plt.close()

print(f"\n--- ConvKAN Done! Stats saved to {STATS_PATH} ---")