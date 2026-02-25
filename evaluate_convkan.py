import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import shared logic
from common import (
    get_convkan_model, get_data_split, prepare_dataset_root,
    CONVKAN_SAVE_PATH, BATCH_SIZE # Updated to use CONVKAN_SAVE_PATH
)

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

# Updated check to use the new constant
if not os.path.exists(CONVKAN_SAVE_PATH):
    print(f"Error: No model weights found at {CONVKAN_SAVE_PATH}. Please run train_convkan.py first.")
    exit()

# 2. LOAD DATA (TEST ONLY)
real_root = prepare_dataset_root()

# We discard the first return (train_dataset) and keep only test_dataset
_, test_dataset = get_data_split(real_root)

# Create loader for test data
loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.dataset.classes # Access underlying classes from subset

print(f"Evaluating on {len(test_dataset)} unseen test images.")

# 3. LOAD MODEL
model = get_convkan_model(device)
# Updated to use CONVKAN_SAVE_PATH
model.load_state_dict(torch.load(CONVKAN_SAVE_PATH, map_location=device))
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
# This will now correctly show results for 2 classes: Parasitized and Uninfected
print(classification_report(y_true, y_pred, target_names=classes))

# 6. PLOT CONFUSION MATRIX
# Because common.py is set to 2 output neurons, cm will correctly be a 2x2 matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title(f"ConvKAN Confusion Matrix (Total Test Images: {len(y_true)})")
plt.show()