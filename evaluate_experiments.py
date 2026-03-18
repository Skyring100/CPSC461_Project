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


def evaluate_model(model, test_loader, device):
    y_true = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    preds_list = np.array(preds_list)
    probs_list = np.array(probs_list)

    # Calculate metrics
    acc = accuracy_score(y_true, preds_list) * 100
    f1 = f1_score(y_true, preds_list, average='binary') * 100
    precision = precision_score(y_true, preds_list, average='binary') * 100
    recall = recall_score(y_true, preds_list, average='binary') * 100
    auc = roc_auc_score(y_true, probs_list[:, 1]) * 100

    # Create confusion matrix
    cm = confusion_matrix(y_true, preds_list)

    return acc, f1, precision, recall, auc, cm


def load_and_evaluate_model(model_name, model_path, get_model_fn, pct_display, test_loader, device):
    if os.path.exists(model_path):
        model = get_model_fn(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        acc, f1, precision, recall, auc, cm = evaluate_model(model, test_loader, device)

        print(f"  {model_name}:")
        print(f"    Accuracy:  {acc:.2f}%")
        print(f"    F1 Score:  {f1:.2f}%")
        print(f"    Precision: {precision:.2f}%")
        print(f"    Recall:    {recall:.2f}%")
        print(f"    AUC:       {auc:.2f}%")

        return {
            'pct': pct_display,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'confusion_matrix': cm
        }
    else:
        print(f"  {model_name}:  NOT FOUND ({model_path})")
        return {
            'pct': pct_display,
            'accuracy': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'auc': None,
            'confusion_matrix': None
        }


def plot_metric_axis(ax, percentages, kan_vals, cnn_vals, ylabel, title):
    ax.plot(percentages, kan_vals, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    ax.plot(percentages, cnn_vals, marker='s', label='CNN Baseline', linewidth=2, markersize=8)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(percentages)


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
    results['convkan'].append(load_and_evaluate_model(
        'ConvKAN', kan_path, get_convkan_model, pct_display, test_loader, device
    ))

    # Load CNN model
    cnn_path = format_experiment_path(CNN_EXPERIMENT_TEMPLATE, percentage)
    results['cnn'].append(load_and_evaluate_model(
        'CNN', cnn_path, get_cnn_model, pct_display, test_loader, device
    ))

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
    plot_metric_axis(axes[0, 0], percentages, kan_accs, cnn_accs, 'Accuracy (%)', 'Accuracy vs Training Data Size')
    # F1 Score plot
    plot_metric_axis(axes[0, 1], percentages, kan_f1s, cnn_f1s, 'F1 Score (%)', 'F1 Score vs Training Data Size')
    # Precision plot
    plot_metric_axis(axes[1, 0], percentages, kan_precisions, cnn_precisions, 'Precision (%)', 'Precision vs Training Data Size')
    # Recall plot
    plot_metric_axis(axes[1, 1], percentages, kan_recalls, cnn_recalls, 'Recall (%)', 'Recall vs Training Data Size')
    
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