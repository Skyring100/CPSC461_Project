import torch
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay
)

from common import (
    get_convkan_model, get_cnn_model, get_data_split, count_parameters,
    prepare_dataset_root, MODELS_ROOT, BATCH_SIZE
)

# --- HELPER: UN-NORMALIZE FOR DISPLAY ---
def denormalize(tensor):
    return tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5

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

DASHBOARD_PATH = os.path.join(NANO_DIR, "nano_benchmark_dashboard.png")
CORRECT_PATH = os.path.join(NANO_DIR, "nano_correct_samples.png")
ERROR_PATH = os.path.join(NANO_DIR, "nano_error_analysis.png")

_, test_dataset = get_data_split(real_root) 
loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = get_root_classes(test_dataset)

# 2. LOAD
kan_model = get_convkan_model(device, version=VERSION)
kan_model.load_state_dict(torch.load(NANO_CONVKAN_PATH, map_location=device))
kan_model.eval()

cnn_model = get_cnn_model(device, version=VERSION)
cnn_model.load_state_dict(torch.load(NANO_CNN_PATH, map_location=device))
cnn_model.eval()

with open(CNN_STATS_PATH, 'r') as f: cnn_stats = json.load(f)
with open(KAN_STATS_PATH, 'r') as f: kan_stats = json.load(f)

# 3. INFERENCE
y_true, kan_preds, cnn_preds, images = [], [], [], []

print("Running inference and collecting visual samples...")
with torch.no_grad():
    for x, y in loader:
        x_dev = x.to(device)
        kan_out = torch.max(kan_model(x_dev), 1)[1].cpu().numpy()
        cnn_out = torch.max(cnn_model(x_dev), 1)[1].cpu().numpy()
        y_true.extend(y.numpy())
        kan_preds.extend(kan_out)
        cnn_preds.extend(cnn_out)
        if len(images) < 500: images.extend(x)

# 4. CATEGORIZE
stored_limit = len(images)
correct_indices = [i for i in range(stored_limit) if kan_preds[i] == y_true[i] and cnn_preds[i] == y_true[i]]
error_indices = [i for i in range(stored_limit) if not (kan_preds[i] == y_true[i] and cnn_preds[i] == y_true[i])]

# 5. METRICS
pos_label = 0 
def calc_metrics(true, pred):
    return [
        f"{accuracy_score(true, pred)*100:.2f}%",
        f"{precision_score(true, pred, pos_label=pos_label)*100:.2f}%",
        f"{recall_score(true, pred, pos_label=pos_label)*100:.2f}%",
        f"{f1_score(true, pred, pos_label=pos_label)*100:.2f}%"
    ]
kan_m = calc_metrics(y_true, kan_preds)
cnn_m = calc_metrics(y_true, cnn_preds)

# --- GRAPHIC 1: DASHBOARD ---
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
fig.suptitle(f"Malaria Detection: {VERSION.upper()} Tier Dashboard", fontsize=20, fontweight='bold', y=0.98)
ax1 = fig.add_subplot(gs[0, 0])
ConfusionMatrixDisplay(confusion_matrix(y_true, kan_preds), display_labels=classes).plot(cmap='Blues', ax=ax1, colorbar=False)
ax1.set_title("Atomic ConvKAN Matrix", pad=15)
ax2 = fig.add_subplot(gs[0, 1])
ConfusionMatrixDisplay(confusion_matrix(y_true, cnn_preds), display_labels=classes).plot(cmap='Reds', ax=ax2, colorbar=False)
ax2.set_title("Lean CNN Matrix", pad=15)
ax3 = fig.add_subplot(gs[1, :]); ax3.axis('off')
table_data = [
    ["Total Parameters", f"{cnn_stats['params']:,}", f"{kan_stats['params']:,}"],
    ["Accuracy", cnn_m[0], kan_m[0]],
    ["Precision", cnn_m[1], kan_m[1]],
    ["Recall", cnn_m[2], kan_m[2]],
    ["F1-Score", cnn_m[3], kan_m[3]],
    ["Peak Training RAM", f"{cnn_stats['peak_memory_mb']:.2f} MB", f"{kan_stats['peak_memory_mb']:.2f} MB"],
    ["Avg Time / Epoch", f"{cnn_stats['avg_time_per_epoch']:.2f} s", f"{kan_stats['avg_time_per_epoch']:.2f} s"]
]
table = ax3.table(cellText=table_data, colLabels=("Metric", "Lean CNN", "Atomic ConvKAN"), loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 2.2)
for (row, col), cell in table.get_celld().items():
    if row == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#333333')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(DASHBOARD_PATH, dpi=300)

# --- GRAPHIC 2: SUCCESS SAMPLES (SPACING FIXED) ---
def plot_success(indices, title, path):
    if not indices: return
    fig, axes = plt.subplots(2, 4, figsize=(16, 11))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    subset = random.sample(indices, min(8, len(indices)))
    for i, idx in enumerate(subset):
        ax = axes[i//4, i%4]
        ax.imshow(denormalize(images[idx]))
        # Use set_title with explicit pad and smaller font to prevent crowding
        ax.set_title(f"TRUE: {classes[y_true[idx]]}\nStatus: Both Correct", 
                     fontsize=10, fontweight='bold', pad=25)
        ax.axis('off')
    
    # hspace=0.6 adds massive vertical space between the rows
    plt.subplots_adjust(hspace=0.6, top=0.85)
    plt.savefig(path, dpi=300); plt.close()

plot_success(correct_indices, "Success Cases: Correctly Identified Cells", CORRECT_PATH)

# --- GRAPHIC 3: ERROR SAMPLES (SPACING FIXED) ---
def plot_errors(indices, title, path):
    if not indices: return
    fig, axes = plt.subplots(2, 4, figsize=(16, 11))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    subset = random.sample(indices, min(8, len(indices)))
    for i, idx in enumerate(subset):
        ax = axes[i//4, i%4]
        ax.imshow(denormalize(images[idx]))
        # Large pad=30 pushes the 3-line title far above the pixels
        ax.set_title(f"TRUE: {classes[y_true[idx]]}\nKAN: {classes[kan_preds[idx]]}\nCNN: {classes[cnn_preds[idx]]}", 
                     fontsize=10, fontweight='bold', pad=30)
        ax.axis('off')
    
    # Manual spacing control
    plt.subplots_adjust(hspace=0.7, top=0.85)
    plt.savefig(path, dpi=300); plt.close()

plot_errors(error_indices, "Failure Cases & Disagreements", ERROR_PATH)

print(f"\nDone! Benchmark images regenerated in: {NANO_DIR}")