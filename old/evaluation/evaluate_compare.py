import torch
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from common import (
    get_convkan_model, get_cnn_model, get_data_split, count_parameters,
    prepare_dataset_root, MODELS_ROOT, BATCH_SIZE
)

def get_root_classes(dataset):
    curr = dataset
    while hasattr(curr, 'dataset'):
        curr = curr.dataset
    return getattr(curr, 'classes', ['Parasitized', 'Uninfected'])

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = "nano"
real_root = prepare_dataset_root()

NANO_DIR = os.path.join(MODELS_ROOT, VERSION)
NANO_CNN_PATH = os.path.join(NANO_DIR, f"malaria_cnn_{VERSION}.pth")
NANO_CONVKAN_PATH = os.path.join(NANO_DIR, f"malaria_convkan_{VERSION}.pth")
CNN_STATS_PATH = os.path.join(NANO_DIR, f"cnn_{VERSION}_stats.json")
KAN_STATS_PATH = os.path.join(NANO_DIR, f"convkan_{VERSION}_stats.json")

_, test_dataset = get_data_split(real_root) 
loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = get_root_classes(test_dataset)

# 2. LOAD MODELS & STATS
print(f"--- Loading {VERSION.upper()} Tier Data ---")

# Load Weights
kan_model = get_convkan_model(device, version=VERSION)
kan_model.load_state_dict(torch.load(NANO_CONVKAN_PATH, map_location=device))
kan_model.eval()

cnn_model = get_cnn_model(device, version=VERSION)
cnn_model.load_state_dict(torch.load(NANO_CNN_PATH, map_location=device))
cnn_model.eval()

# Load JSON Stats
with open(CNN_STATS_PATH, 'r') as f: cnn_stats = json.load(f)
with open(KAN_STATS_PATH, 'r') as f: kan_stats = json.load(f)

# 3. RUN INFERENCE TEST
y_true, kan_preds, cnn_preds = [], [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_true.extend(y.cpu().numpy())
        kan_preds.extend(torch.max(kan_model(x), 1)[1].cpu().numpy())
        cnn_preds.extend(torch.max(cnn_model(x), 1)[1].cpu().numpy())

# 4. DATA CALCULATIONS
kan_acc = accuracy_score(y_true, kan_preds) * 100
cnn_acc = accuracy_score(y_true, cnn_preds) * 100

# Efficiency Score: How much accuracy do you get per 1,000 parameters?
kan_eff = kan_acc / (kan_stats['params'] / 1000)
cnn_eff = cnn_acc / (cnn_stats['params'] / 1000)

# 5. THE ULTIMATE RESOURCE COMPARISON TABLE
print("\n" + "="*65)
print(f"{'FINAL NANO-TIER BENCHMARK REPORT':^65}")
print("="*65)
print(f"{'Metric':<22} | {'Wide CNN':<18} | {'Thin ConvKAN':<18}")
print("-" * 65)
print(f"{'Total Parameters':<22} | {cnn_stats['params']:<18,} | {kan_stats['params']:<18,}")
print(f"{'Test Accuracy':<22} | {cnn_acc:<17.2f}% | {kan_acc:<17.2f}%")
print(f"{'Acc / 1k Params':<22} | {cnn_eff:<18.2f} | {kan_eff:<18.2f}")
print("-" * 65)
print(f"{'Peak Training RAM':<22} | {cnn_stats['peak_memory_mb']:>14.2f} MB | {kan_stats['peak_memory_mb']:>14.2f} MB")
print(f"{'Total Training Time':<22} | {cnn_stats['total_time_sec']:>14.2f} s  | {kan_stats['total_time_sec']:>14.2f} s")
print(f"{'Avg Time / Epoch':<22} | {cnn_stats['avg_time_per_epoch']:>14.2f} s  | {kan_stats['avg_time_per_epoch']:>14.2f} s")
print("-" * 65)

# Winner Logic based on Efficiency Score
if kan_eff > cnn_eff:
    print(f"OVERALL WINNER: ConvKAN ({kan_eff/cnn_eff:.2f}x more parameter-efficient)")
else:
    print(f"OVERALL WINNER: CNN ({cnn_eff/kan_eff:.2f}x more parameter-efficient)")
print("="*65)

# 6. PLOT
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_true, kan_preds), display_labels=classes).plot(cmap='Blues', ax=axes[0], colorbar=False)
axes[0].set_title(f"Nano ConvKAN ({kan_acc:.1f}%)")
ConfusionMatrixDisplay(confusion_matrix(y_true, cnn_preds), display_labels=classes).plot(cmap='Reds', ax=axes[1], colorbar=False)
axes[1].set_title(f"Nano CNN ({cnn_acc:.1f}%)")
plt.tight_layout(); plt.show()