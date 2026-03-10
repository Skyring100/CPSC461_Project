import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
import numpy as np

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

# Create output directory for confusion matrices
os.makedirs('confusion_matrices', exist_ok=True)

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
        kan_probs = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = kan_model(x)
                probs = torch.softmax(outputs, 1)
                _, preds = torch.max(outputs, 1)
                y_true.extend(y.cpu().numpy())
                kan_preds.extend(preds.cpu().numpy())
                kan_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(y_true)
        kan_preds = np.array(kan_preds)
        kan_probs = np.array(kan_probs)
        
        # Calculate metrics
        kan_acc = accuracy_score(y_true, kan_preds) * 100
        kan_f1 = f1_score(y_true, kan_preds, average='binary') * 100
        kan_precision = precision_score(y_true, kan_preds, average='binary') * 100
        kan_recall = recall_score(y_true, kan_preds, average='binary') * 100
        kan_auc = roc_auc_score(y_true, kan_probs[:, 1]) * 100
        
        # Create confusion matrix
        kan_cm = confusion_matrix(y_true, kan_preds)
        
        results['convkan'].append({
            'pct': pct_display,
            'accuracy': kan_acc,
            'f1': kan_f1,
            'precision': kan_precision,
            'recall': kan_recall,
            'auc': kan_auc,
            'confusion_matrix': kan_cm
        })
        
        print(f"  ConvKAN:")
        print(f"    Accuracy:  {kan_acc:.2f}%")
        print(f"    F1 Score:  {kan_f1:.2f}%")
        print(f"    Precision: {kan_precision:.2f}%")
        print(f"    Recall:    {kan_recall:.2f}%")
        print(f"    AUC:       {kan_auc:.2f}%")
    else:
        print(f"  ConvKAN:  NOT FOUND ({kan_path})")
        results['convkan'].append({
            'pct': pct_display,
            'accuracy': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'auc': None,
            'confusion_matrix': None
        })
    
    # Load CNN model
    cnn_path = format_experiment_path(CNN_EXPERIMENT_TEMPLATE, percentage)
    if os.path.exists(cnn_path):
        cnn_model = get_cnn_model(device)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.eval()
        
        # Evaluate CNN
        y_true = []
        cnn_preds = []
        cnn_probs = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = cnn_model(x)
                probs = torch.softmax(outputs, 1)
                _, preds = torch.max(outputs, 1)
                y_true.extend(y.cpu().numpy())
                cnn_preds.extend(preds.cpu().numpy())
                cnn_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(y_true)
        cnn_preds = np.array(cnn_preds)
        cnn_probs = np.array(cnn_probs)
        
        # Calculate metrics
        cnn_acc = accuracy_score(y_true, cnn_preds) * 100
        cnn_f1 = f1_score(y_true, cnn_preds, average='binary') * 100
        cnn_precision = precision_score(y_true, cnn_preds, average='binary') * 100
        cnn_recall = recall_score(y_true, cnn_preds, average='binary') * 100
        cnn_auc = roc_auc_score(y_true, cnn_probs[:, 1]) * 100
        
        # Create confusion matrix
        cnn_cm = confusion_matrix(y_true, cnn_preds)
        
        results['cnn'].append({
            'pct': pct_display,
            'accuracy': cnn_acc,
            'f1': cnn_f1,
            'precision': cnn_precision,
            'recall': cnn_recall,
            'auc': cnn_auc,
            'confusion_matrix': cnn_cm
        })
        
        print(f"  CNN:")
        print(f"    Accuracy:  {cnn_acc:.2f}%")
        print(f"    F1 Score:  {cnn_f1:.2f}%")
        print(f"    Precision: {cnn_precision:.2f}%")
        print(f"    Recall:    {cnn_recall:.2f}%")
        print(f"    AUC:       {cnn_auc:.2f}%")
    else:
        print(f"  CNN:      NOT FOUND ({cnn_path})")
        results['cnn'].append({
            'pct': pct_display,
            'accuracy': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'auc': None,
            'confusion_matrix': None
        })

# 3. PLOT CONFUSION MATRICES
print(f"\n{'='*70}")
print("GENERATING CONFUSION MATRICES")
print(f"{'='*70}")

for i, pct in enumerate(PERCENTAGES):
    pct_display = int(pct * 100)
    
    kan_result = results['convkan'][i]
    cnn_result = results['cnn'][i]
    
    if kan_result['confusion_matrix'] is not None and cnn_result['confusion_matrix'] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ConvKAN confusion matrix
        disp_kan = ConfusionMatrixDisplay(
            confusion_matrix=kan_result['confusion_matrix'],
            display_labels=classes
        )
        disp_kan.plot(cmap='Blues', ax=axes[0], colorbar=False)
        axes[0].set_title(f"ConvKAN ({pct_display}% data)\nF1: {kan_result['f1']:.2f}% | Acc: {kan_result['accuracy']:.2f}%")
        
        # CNN confusion matrix
        disp_cnn = ConfusionMatrixDisplay(
            confusion_matrix=cnn_result['confusion_matrix'],
            display_labels=classes
        )
        disp_cnn.plot(cmap='Reds', ax=axes[1], colorbar=False)
        axes[1].set_title(f"CNN ({pct_display}% data)\nF1: {cnn_result['f1']:.2f}% | Acc: {cnn_result['accuracy']:.2f}%")
        
        plt.tight_layout()
        save_path = f'confusion_matrices/confusion_matrix_{pct_display}pct.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

# 4. PRINT DETAILED RESULTS TABLE
print(f"\n{'='*70}")
print("DETAILED COMPARISON TABLE")
print(f"{'='*70}\n")

# Print ConvKAN metrics
print("ConvKAN METRICS:")
print(f"{'Data %':<10} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'AUC':<12}")
print("-" * 70)
for result in results['convkan']:
    if result['accuracy'] is not None:
        print(f"{result['pct']:<10} {result['accuracy']:<12.2f} {result['f1']:<12.2f} {result['precision']:<12.2f} {result['recall']:<12.2f} {result['auc']:<12.2f}")
    else:
        print(f"{result['pct']:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("\n" + "="*70 + "\n")

# Print CNN metrics
print("CNN METRICS:")
print(f"{'Data %':<10} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'AUC':<12}")
print("-" * 70)
for result in results['cnn']:
    if result['accuracy'] is not None:
        print(f"{result['pct']:<10} {result['accuracy']:<12.2f} {result['f1']:<12.2f} {result['precision']:<12.2f} {result['recall']:<12.2f} {result['auc']:<12.2f}")
    else:
        print(f"{result['pct']:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("\n" + "="*70 + "\n")

# Print comparison
print("WINNER BY METRIC:")
print(f"{'Data %':<10} {'Accuracy':<15} {'F1 Score':<15} {'Precision':<15} {'Recall':<15}")
print("-" * 70)
for i, pct in enumerate(PERCENTAGES):
    kan = results['convkan'][i]
    cnn = results['cnn'][i]
    
    if kan['accuracy'] is not None and cnn['accuracy'] is not None:
        acc_winner = "ConvKAN" if kan['accuracy'] > cnn['accuracy'] else ("CNN" if cnn['accuracy'] > kan['accuracy'] else "Tie")
        f1_winner = "ConvKAN" if kan['f1'] > cnn['f1'] else ("CNN" if cnn['f1'] > kan['f1'] else "Tie")
        prec_winner = "ConvKAN" if kan['precision'] > cnn['precision'] else ("CNN" if cnn['precision'] > kan['precision'] else "Tie")
        rec_winner = "ConvKAN" if kan['recall'] > cnn['recall'] else ("CNN" if cnn['recall'] > kan['recall'] else "Tie")
        
        print(f"{int(pct*100):<10} {acc_winner:<15} {f1_winner:<15} {prec_winner:<15} {rec_winner:<15}")

# 5. CREATE COMPARISON PLOTS
print(f"\n{'='*70}")
print("GENERATING COMPARISON PLOTS")
print(f"{'='*70}")

percentages = [int(p * 100) for p in PERCENTAGES]
kan_accs = [r['accuracy'] for r in results['convkan'] if r['accuracy'] is not None]
cnn_accs = [r['accuracy'] for r in results['cnn'] if r['accuracy'] is not None]
kan_f1s = [r['f1'] for r in results['convkan'] if r['f1'] is not None]
cnn_f1s = [r['f1'] for r in results['cnn'] if r['f1'] is not None]
kan_precisions = [r['precision'] for r in results['convkan'] if r['precision'] is not None]
cnn_precisions = [r['precision'] for r in results['cnn'] if r['precision'] is not None]
kan_recalls = [r['recall'] for r in results['convkan'] if r['recall'] is not None]
cnn_recalls = [r['recall'] for r in results['cnn'] if r['recall'] is not None]
kan_aucs = [r['auc'] for r in results['convkan'] if r['auc'] is not None]
cnn_aucs = [r['auc'] for r in results['cnn'] if r['auc'] is not None]

if kan_accs and cnn_accs:
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy plot
    axes[0, 0].plot(percentages, kan_accs, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    axes[0, 0].plot(percentages, cnn_accs, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Training Data Percentage (%)', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_title('Accuracy vs Training Data Size', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].set_xticks(percentages)
    
    # F1 Score plot
    axes[0, 1].plot(percentages, kan_f1s, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    axes[0, 1].plot(percentages, cnn_f1s, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Training Data Percentage (%)', fontsize=11)
    axes[0, 1].set_ylabel('F1 Score (%)', fontsize=11)
    axes[0, 1].set_title('F1 Score vs Training Data Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].set_xticks(percentages)
    
    # Precision plot
    axes[1, 0].plot(percentages, kan_precisions, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    axes[1, 0].plot(percentages, cnn_precisions, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Training Data Percentage (%)', fontsize=11)
    axes[1, 0].set_ylabel('Precision (%)', fontsize=11)
    axes[1, 0].set_title('Precision vs Training Data Size', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_xticks(percentages)
    
    # Recall plot
    axes[1, 1].plot(percentages, kan_recalls, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    axes[1, 1].plot(percentages, cnn_recalls, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Training Data Percentage (%)', fontsize=11)
    axes[1, 1].set_ylabel('Recall (%)', fontsize=11)
    axes[1, 1].set_title('Recall vs Training Data Size', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_xticks(percentages)
    
    fig.suptitle('Model Performance Comparison - Multiple Metrics\n(Evaluated on held-out 10% test set)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('experiment_comparison_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved: experiment_comparison_metrics.png")
    plt.show()
    
    # AUC plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(percentages, kan_aucs, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    ax.plot(percentages, cnn_aucs, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax.set_ylabel('AUC (%)', fontsize=12)
    ax.set_title('AUC-ROC vs Training Data Size\n(Evaluated on held-out 10% test set)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(percentages)
    fig.tight_layout()
    plt.savefig('experiment_auc_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: experiment_auc_comparison.png")
    plt.show()

print(f"\n{'='*70}")
print("Evaluation complete!")
print(f"{'='*70}")
print("\nGenerated files:")
print("  - confusion_matrices/ (folder with CM for each experiment)")
print("  - experiment_comparison_metrics.png (Accuracy, F1, Precision, Recall)")
print("  - experiment_auc_comparison.png (AUC-ROC curves)")
