"""
Evaluation script for overfitting experiments.

This script evaluates already-trained models on the held-out test set
and generates comparison plots and summary tables.

It assumes models have already been trained using run_overfitting_experiments.py
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from common import (
    get_cnn_model, get_convkan_model, get_test_set,
    prepare_dataset_root, BATCH_SIZE, get_overfit_sweep_path
)

# Sample sizes that were trained
SAMPLE_SIZES = [25, 50, 100, 200, 300, 400, 500]

def evaluate_on_test_set(model, test_loader, device):
    """Evaluate model on test set and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating overfitting experiments on: {device}")
    print(f"Sample sizes: {SAMPLE_SIZES}\n")
    
    # Prepare dataset
    real_root = prepare_dataset_root()
    
    # Load test set
    test_dataset = get_test_set(real_root)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_dataset)} images\n")
    
    # Results storage
    results = {
        'cnn': {},
        'convkan': {}
    }
    
    print("="*80)
    print("LOADING AND EVALUATING TRAINED MODELS")
    print("="*80)
    
    # Evaluate models for each sample size
    for samples_per_class in SAMPLE_SIZES:
        print(f"\nEvaluating models trained on {samples_per_class} samples/class...")
        
        # Evaluate CNN
        cnn_path = get_overfit_sweep_path("cnn", samples_per_class)
        if os.path.exists(cnn_path):
            cnn_model = get_cnn_model(device)
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
            cnn_test_acc = evaluate_on_test_set(cnn_model, test_loader, device)
            results['cnn'][samples_per_class] = cnn_test_acc
            print(f"  CNN:     {cnn_test_acc:.2f}%")
        else:
            print(f"  CNN:     NOT FOUND ({cnn_path})")
            results['cnn'][samples_per_class] = None
        
        # Evaluate ConvKAN
        kan_path = get_overfit_sweep_path("convkan", samples_per_class)
        if os.path.exists(kan_path):
            kan_model = get_convkan_model(device)
            kan_model.load_state_dict(torch.load(kan_path, map_location=device))
            kan_test_acc = evaluate_on_test_set(kan_model, test_loader, device)
            results['convkan'][samples_per_class] = kan_test_acc
            print(f"  ConvKAN: {kan_test_acc:.2f}%")
        else:
            print(f"  ConvKAN: NOT FOUND ({kan_path})")
            results['convkan'][samples_per_class] = None
    
    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    # Create output directory
    os.makedirs('overfitting_experiments', exist_ok=True)
    
    # Plot: Test accuracy vs sample size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cnn_test_accs = [results['cnn'][s] for s in SAMPLE_SIZES if results['cnn'][s] is not None]
    kan_test_accs = [results['convkan'][s] for s in SAMPLE_SIZES if results['convkan'][s] is not None]
    valid_sizes = [s for s in SAMPLE_SIZES if results['cnn'][s] is not None]
    
    if cnn_test_accs and kan_test_accs:
        ax.plot(valid_sizes, cnn_test_accs, marker='s', label='CNN', linewidth=2, markersize=8)
        ax.plot(valid_sizes, kan_test_accs, marker='o', label='ConvKAN', linewidth=2, markersize=8)
        
        ax.set_xlabel('Training Samples per Class', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Overfitting Experiments: Test Accuracy vs Training Data Size\n(Evaluated on 10% held-out test set)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xscale('log')
        ax.set_xticks(valid_sizes)
        ax.set_xticklabels(valid_sizes)
        
        plt.tight_layout()
        plt.savefig('overfitting_experiments/test_accuracy_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: overfitting_experiments/test_accuracy_comparison.png")
        plt.show()
    else:
        print("Warning: Could not plot - missing model results")
    
    # Summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Samples/Class':<15} {'CNN Acc':<12} {'ConvKAN Acc':<12} {'Winner':<15} {'Difference':<12}")
    print("-" * 80)
    
    for samples in SAMPLE_SIZES:
        cnn_acc = results['cnn'][samples]
        kan_acc = results['convkan'][samples]
        
        if cnn_acc is not None and kan_acc is not None:
            winner = "ConvKAN" if kan_acc > cnn_acc else ("CNN" if cnn_acc > kan_acc else "Tie")
            difference = abs(kan_acc - cnn_acc)
            print(f"{samples:<15} {cnn_acc:<12.2f} {kan_acc:<12.2f} {winner:<15} {difference:<12.2f}%")
        else:
            print(f"{samples:<15} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    # Overall winner
    print("\n" + "-" * 80)
    cnn_wins = sum(1 for s in SAMPLE_SIZES if results['cnn'][s] is not None and 
                   results['convkan'][s] is not None and results['cnn'][s] > results['convkan'][s])
    kan_wins = sum(1 for s in SAMPLE_SIZES if results['cnn'][s] is not None and 
                   results['convkan'][s] is not None and results['convkan'][s] > results['cnn'][s])
    
    print(f"Overall: CNN wins {cnn_wins} experiments, ConvKAN wins {kan_wins} experiments")
    
    if cnn_wins > kan_wins:
        print("✓ CNN is better overall on limited data")
    elif kan_wins > cnn_wins:
        print("✓ ConvKAN is better overall on limited data")
    else:
        print("✓ Both models tied overall")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
