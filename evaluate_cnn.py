import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import shared logic from common.py
from common import (
    get_cnn_model, get_data_split, prepare_dataset_root,
    CNN_SAVE_PATH, BATCH_SIZE
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating CNN Baseline on device: {device}")

# Ensure weights exist before trying to load them
if not os.path.exists(CNN_SAVE_PATH):
    print(f"Error: No CNN weights found at {CNN_SAVE_PATH}. Please run train_cnn.py first.")
    exit()

# 2. LOAD DATA
real_root = prepare_dataset_root()

# Discard train_dataset, keep test_dataset
_, test_dataset = get_data_split(real_root)

loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.dataset.classes

print(f"Evaluating on {len(test_dataset)} unseen test images.")

# 3. LOAD CNN MODEL
model = get_cnn_model(device)
model.load_state_dict(torch.load(CNN_SAVE_PATH, map_location=device))
model.eval()

# 4. RUN INFERENCE
y_true = []
y_pred = []

print("Running CNN predictions...")
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 5. PRINT CLASSIFICATION REPORT
print("\n" + "="*40)
print("           CNN TEST RESULTS")
print("="*40)
print(classification_report(y_true, y_pred, target_names=classes))

# 6. PLOT CONFUSION MATRIX
# This will be a 2x2 matrix since we updated common.py to 2 output classes
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Reds', ax=ax) # Using Reds to distinguish from ConvKAN's Blue
plt.title(f"CNN Confusion Matrix (Total Test Images: {len(y_true)})")
plt.show()