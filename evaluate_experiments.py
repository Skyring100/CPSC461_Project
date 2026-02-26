import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Import shared logic
from common import (
    get_convkan_model, get_cnn_model, get_test_set,
    prepare_dataset_root, CONVKAN_EXPERIMENT_TEMPLATE, CNN_EXPERIMENT_TEMPLATE, 
    format_experiment_path, BATCH_SIZE
)

# 1. SETUP & DATA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating experiments on: {device}\n")

real_root = prepare_dataset_root()

# Load the held-out 10% test set (used for all experiments)
test_dataset = get_test_set(real_root)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.dataset.classes

print(f"Using held-out test set with {len(test_dataset)} images")
print(f"Classes: {classes}\n")

# Data percentages (must match training experiments)
PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 2. LOAD ALL MODELS AND EVALUATE
results = {
    'convkan': [],
    'cnn': []
}

print("="*70)
print("EVALUATING EXPERIMENTS ON HELD-OUT TEST SET (10% of all data)")
print("="*70)

for percentage in PERCENTAGES:
    pct_display = int(percentage * 100)
    print(f"\nExperiment: {pct_display}% training data")
    print("-" * 70)
    
    # Load ConvKAN model
    kan_path = format_experiment_path(CONVKAN_EXPERIMENT_TEMPLATE, percentage)
    if os.path.exists(kan_path):
        kan_model = get_convkan_model(device)
        kan_model.load_state_dict(torch.load(kan_path, map_location=device))
        kan_model.eval()
        
        # Evaluate ConvKAN
        y_true = []
        kan_preds = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = kan_model(x)
                _, preds = torch.max(outputs, 1)
                y_true.extend(y.cpu().numpy())
                kan_preds.extend(preds.cpu().numpy())
        
        kan_acc = accuracy_score(y_true, kan_preds) * 100
        results['convkan'].append((pct_display, kan_acc))
        print(f"  ConvKAN:  {kan_acc:.2f}%")
    else:
        print(f"  ConvKAN:  NOT FOUND ({kan_path})")
        results['convkan'].append((pct_display, None))
    
    # Load CNN model
    cnn_path = format_experiment_path(CNN_EXPERIMENT_TEMPLATE, percentage)
    if os.path.exists(cnn_path):
        cnn_model = get_cnn_model(device)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.eval()
        
        # Evaluate CNN
        y_true = []
        cnn_preds = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = cnn_model(x)
                _, preds = torch.max(outputs, 1)
                y_true.extend(y.cpu().numpy())
                cnn_preds.extend(preds.cpu().numpy())
        
        cnn_acc = accuracy_score(y_true, cnn_preds) * 100
        results['cnn'].append((pct_display, cnn_acc))
        print(f"  CNN:      {cnn_acc:.2f}%")
    else:
        print(f"  CNN:      NOT FOUND ({cnn_path})")
        results['cnn'].append((pct_display, None))

# 3. PLOT RESULTS
print(f"\n{'='*70}")
print("SUMMARY OF RESULTS")
print(f"{'='*70}\n")

# Print table
print(f"{'Data %':<10} {'ConvKAN':<15} {'CNN':<15} {'Winner':<15}")
print("-" * 55)

for i, pct in enumerate(PERCENTAGES):
    pct_display = int(pct * 100)
    kan_acc = results['convkan'][i][1]
    cnn_acc = results['cnn'][i][1]
    
    if kan_acc is not None and cnn_acc is not None:
        kan_str = f"{kan_acc:.2f}%"
        cnn_str = f"{cnn_acc:.2f}%"
        if kan_acc > cnn_acc:
            winner = "ConvKAN"
        elif cnn_acc > kan_acc:
            winner = "CNN"
        else:
            winner = "Tie"
        print(f"{pct_display:<10} {kan_str:<15} {cnn_str:<15} {winner:<15}")
    else:
        print(f"{pct_display:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

# Create comparison plot
fig, ax = plt.subplots(figsize=(12, 6))

percentages = [int(p * 100) for p in PERCENTAGES]
kan_accs = [acc for _, acc in results['convkan'] if acc is not None]
cnn_accs = [acc for _, acc in results['cnn'] if acc is not None]

if kan_accs and cnn_accs:
    ax.plot(percentages, kan_accs, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    ax.plot(percentages, cnn_accs, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    
    ax.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance vs Training Data Size\n(Evaluated on held-out 10% test set)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(percentages)
    ax.set_ylim([min(min(kan_accs), min(cnn_accs)) - 5, 105])
    
    fig.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to 'experiment_comparison.png'")
    plt.show()

print(f"\n{'='*70}")
print("Evaluation complete!")
print(f"{'='*70}")
