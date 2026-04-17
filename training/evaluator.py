import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def run_final_evaluation(model, loader, device) -> dict:
    """Evaluate model on loader and return a metrics dict"""
    model.eval()
    all_preds, all_labels = [], []

    # No gradient calculations needed for evaluation
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.view(-1).long().to(device)
            _, predicted = torch.max(model(x), dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def print_final_report(model_name: str, results: dict):
    """Print a formatted summary of results."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"FINAL RESULTS FOR {model_name.upper()}")
    print(sep)
    print(f"Total Parameters:      {results['num_params']:,}")
    print(f"Overall Accuracy:      {results['accuracy']:.2f}%")
    print(f"Precision (weighted):  {results['precision']:.4f}")
    print(f"Recall (weighted):     {results['recall']:.4f}")
    print(f"F1 Score (weighted):   {results['f1']:.4f}")
    print(f"Peak Training RAM:     {results['peak_ram']:.2f} GB")
    print(f"Peak Training VRAM:    {results['peak_vram']:.2f} GB")
    print(f"Total Training Time:   {results['total_time']:.2f}s")
    print(f"Avg Time per Epoch:    {results['avg_epoch_time']:.2f}s")
    print(f"Total Epochs Trained:  {results['num_epochs']}")
    print(f"{sep}\n")