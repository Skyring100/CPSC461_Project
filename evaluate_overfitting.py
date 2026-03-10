import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score
)
import numpy as np

# Import shared logic
from common import (
    get_convkan_model, get_cnn_model, get_test_set,
    prepare_dataset_root, CONVKAN_OVERFITTING_PATH, CNN_OVERFITTING_PATH,
    BATCH_SIZE
)

# 1. SETUP & DATA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating overfitting experiment on: {device}\n")

real_root = prepare_dataset_root()

# Load the held-out 10% test set (used for evaluation)
test_dataset = get_test_set(real_root)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.dataset.classes

print(f"Using held-out test set with {len(test_dataset)} images")
print(f"Classes: {classes}\n")

print("="*80)
print("OVERFITTING EXPERIMENT - EVALUATION ON HELD-OUT TEST SET")
print("(Models trained on only 250 samples per class)")
print("="*80)

# 2. FUNCTION TO EVALUATE MODEL
def evaluate_model(model_path, model_name):
    """Evaluate a model on the test set and return metrics."""
    if not os.path.exists(model_path):
        print(f"{model_name}: Model not found at {model_path}")
        return None
    
    print(f"\nEvaluating {model_name}...")
    
    if "convkan" in model_name.lower():
        model = get_convkan_model(device)
    else:
        model = get_cnn_model(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    y_true = []
    y_preds = []
    y_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    y_probs = np.array(y_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_preds) * 100,
        'f1': f1_score(y_true, y_preds, average='binary') * 100,
        'precision': precision_score(y_true, y_preds, average='binary') * 100,
        'recall': recall_score(y_true, y_preds, average='binary') * 100,
        'auc': roc_auc_score(y_true, y_probs[:, 1]) * 100,
        'confusion_matrix': confusion_matrix(y_true, y_preds),
        'y_true': y_true,
        'y_preds': y_preds
    }
    
    return metrics

# 3. EVALUATE BOTH MODELS
results = {}

# Evaluate CNN
cnn_metrics = evaluate_model(CNN_OVERFITTING_PATH, "CNN Overfitting")
if cnn_metrics:
    results['cnn'] = cnn_metrics
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.2f}%")
    print(f"  F1 Score:  {cnn_metrics['f1']:.2f}%")
    print(f"  Precision: {cnn_metrics['precision']:.2f}%")
    print(f"  Recall:    {cnn_metrics['recall']:.2f}%")
    print(f"  AUC:       {cnn_metrics['auc']:.2f}%")

# Evaluate ConvKAN
kan_metrics = evaluate_model(CONVKAN_OVERFITTING_PATH, "ConvKAN Overfitting")
if kan_metrics:
    results['convkan'] = kan_metrics
    print(f"  Accuracy:  {kan_metrics['accuracy']:.2f}%")
    print(f"  F1 Score:  {kan_metrics['f1']:.2f}%")
    print(f"  Precision: {kan_metrics['precision']:.2f}%")
    print(f"  Recall:    {kan_metrics['recall']:.2f}%")
    print(f"  AUC:       {kan_metrics['auc']:.2f}%")

# 4. PRINT COMPARISON
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}\n")

if len(results) == 2:
    kan = results['convkan']
    cnn = results['cnn']
    
    print(f"{'Metric':<15} {'ConvKAN':<15} {'CNN':<15} {'Winner':<15}")
    print("-" * 60)
    
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('F1 Score', 'f1'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('AUC', 'auc')
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        kan_val = kan[metric_key]
        cnn_val = cnn[metric_key]
        winner = "ConvKAN" if kan_val > cnn_val else ("CNN" if cnn_val > kan_val else "Tie")
        print(f"{metric_name:<15} {kan_val:<15.2f} {cnn_val:<15.2f} {winner:<15}")

# 5. PLOT CONFUSION MATRICES
if len(results) == 2:
    os.makedirs('confusion_matrices', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    kan = results['convkan']
    cnn = results['cnn']
    
    # ConvKAN confusion matrix
    disp_kan = ConfusionMatrixDisplay(
        confusion_matrix=kan['confusion_matrix'],
        display_labels=classes
    )
    disp_kan.plot(cmap='Blues', ax=axes[0], colorbar=False)
    axes[0].set_title(f"ConvKAN (Overfitting)\nF1: {kan['f1']:.2f}% | Acc: {kan['accuracy']:.2f}%")
    
    # CNN confusion matrix
    disp_cnn = ConfusionMatrixDisplay(
        confusion_matrix=cnn['confusion_matrix'],
        display_labels=classes
    )
    disp_cnn.plot(cmap='Reds', ax=axes[1], colorbar=False)
    axes[1].set_title(f"CNN (Overfitting)\nF1: {cnn['f1']:.2f}% | Acc: {cnn['accuracy']:.2f}%")
    
    plt.tight_layout()
    save_path = 'confusion_matrices/confusion_matrix_overfitting.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrices saved to {save_path}")
    plt.show()

print(f"\n{'='*80}")
print("Overfitting evaluation complete!")
print(f"{'='*80}")
