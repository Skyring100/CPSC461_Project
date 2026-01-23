import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from common import (
    get_model, get_cnn_model, find_data_root, get_data_split, 
    DATASET_PATH, MODEL_SAVE_PATH, BATCH_SIZE
)

CNN_SAVE_PATH = "malaria_cnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD DATA
real_root = find_data_root(DATASET_PATH)
_, test_dataset = get_data_split(real_root) # Test set only
loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.dataset.classes

# 2. LOAD MODELS
print("Loading ConvKAN...")
kan_model = get_model(device)
kan_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
kan_model.eval()

print("Loading CNN...")
cnn_model = get_cnn_model(device)
if os.path.exists(CNN_SAVE_PATH):
    cnn_model.load_state_dict(torch.load(CNN_SAVE_PATH, map_location=device))
    cnn_model.eval()
else:
    print("Warning: CNN weights not found! Run train_cnn.py first.")
    exit()

# 3. RUN BATTLE
y_true = []
kan_preds = []
cnn_preds = []

print(f"\nComparing models on {len(test_dataset)} images...")

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        y_true.extend(y.cpu().numpy())
        
        # ConvKAN
        _, k_pred = torch.max(kan_model(x), 1)
        kan_preds.extend(k_pred.cpu().numpy())
        
        # CNN
        _, c_pred = torch.max(cnn_model(x), 1)
        cnn_preds.extend(c_pred.cpu().numpy())

# 4. TEXT RESULTS
kan_acc = accuracy_score(y_true, kan_preds) * 100
cnn_acc = accuracy_score(y_true, cnn_preds) * 100

print("\n" + "="*30)
print("      FINAL RESULTS")
print("="*30)
print(f"ConvKAN Accuracy: {kan_acc:.2f}%")
print(f"CNN Accuracy:     {cnn_acc:.2f}%")
print("="*30)

if kan_acc > cnn_acc:
    print("ConvKAN wins!")
else:
    print("CNN wins!")

# 5. PLOT CONFUSION MATRICES
print("\nPlotting Confusion Matrices...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot ConvKAN
cm_kan = confusion_matrix(y_true, kan_preds)
disp_kan = ConfusionMatrixDisplay(confusion_matrix=cm_kan, display_labels=classes)
disp_kan.plot(cmap='Blues', ax=axes[0], colorbar=False)
axes[0].set_title(f"ConvKAN (Acc: {kan_acc:.1f}%)")

# Plot CNN
cm_cnn = confusion_matrix(y_true, cnn_preds)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=classes)
disp_cnn.plot(cmap='Reds', ax=axes[1], colorbar=False)
axes[1].set_title(f"CNN Baseline (Acc: {cnn_acc:.1f}%)")

plt.tight_layout()
plt.show()