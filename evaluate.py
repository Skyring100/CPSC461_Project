import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import shared logic
from common import (
    get_model, find_data_root, get_data_split, 
    DATASET_PATH, MODEL_SAVE_PATH, BATCH_SIZE
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

if not os.path.exists(MODEL_SAVE_PATH):
    print("Error: No model weights found. Please run main.py first.")
    exit()

# 2. LOAD DATA (TEST ONLY)
real_root = find_data_root(DATASET_PATH)

# We discard the first return (train_dataset) and keep only test_dataset
_, test_dataset = get_data_split(real_root)

# Create loader for test data
loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.dataset.classes # Access underlying classes from subset

print(f"Evaluating on {len(test_dataset)} unseen test images.")

# 3. LOAD MODEL
model = get_model(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# 4. RUN INFERENCE
y_true = []
y_pred = []

print("Running predictions...")
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 5. REPORT
print("\n" + "="*40)
print("           TEST RESULTS")
print("="*40)
print(classification_report(y_true, y_pred, target_names=classes))

# 6. PLOT CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title(f"Confusion Matrix (Total Test Images: {len(y_true)})")
plt.show()