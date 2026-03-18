"""
Master script to run overfitting experiments with various sample sizes.

This script trains CNN and ConvKAN models on limited data to observe
how they perform with different amounts of training data.

Sample sizes tested per class: 25, 50, 100, 200, 300, 400, 500, 1000
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

from common import (
    get_cnn_model, get_convkan_model, get_small_fixed_dataset, get_test_set,
    prepare_dataset_root, BATCH_SIZE, get_overfit_sweep_path, ensure_parent_dir
)

# Sample sizes to test (samples per class)
SAMPLE_SIZES = [25, 50, 100, 200, 300, 400, 500]
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 5

def get_batch_size(samples_per_class):
    """Determine appropriate batch size based on dataset size."""
    total_samples = samples_per_class * 2
    if total_samples <= 100:
        return 8
    elif total_samples <= 400:
        return 16
    else:
        return 32

def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_name, samples_per_class):
    """Train a model with early stopping and return training history."""
    print(f"\nTraining {model_name} on {samples_per_class} samples/class...")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    epoch = 0
    
    while epochs_no_improve < PATIENCE and epoch < MAX_EPOCHS:
        epoch += 1
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(y_hat, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(y_hat, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Check for overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epoch % 5 == 0 or epochs_no_improve == 1:
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f}/{train_acc:.1f}% | Val: {val_loss:.4f}/{val_acc:.1f}% | No improve: {epochs_no_improve}/{PATIENCE}")
    
    print(f"  Training stopped at epoch {epoch}. Best epoch: {best_epoch}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'total_epochs': epoch
    }

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
    print(f"Running overfitting experiments on: {device}")
    print(f"Sample sizes per class: {SAMPLE_SIZES}\n")
    
    # Prepare dataset
    real_root = prepare_dataset_root()
    
    # Load test set (same for all experiments)
    test_dataset = get_test_set(real_root)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_dataset)} images\n")
    
    # Results storage
    results = {
        'cnn': defaultdict(dict),
        'convkan': defaultdict(dict)
    }
    
    # Run experiments for each sample size
    for samples_per_class in SAMPLE_SIZES:
        print("\n" + "="*80)
        print(f"EXPERIMENT: {samples_per_class} samples per class ({samples_per_class*2} total)")
        print("="*80)
        
        # Load dataset for this sample size
        train_dataset, val_dataset = get_small_fixed_dataset(real_root, samples_per_class=samples_per_class)
        
        batch_size = get_batch_size(samples_per_class)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training: {len(train_dataset)} samples | Validation: {len(val_dataset)} samples")
        print(f"Batch size: {batch_size}")
        
        # Train CNN
        cnn_model = get_cnn_model(device, version="android")
        class_weights = torch.tensor([1.0, 1.0], device=device)
        cnn_criterion = nn.CrossEntropyLoss(weight=class_weights)
        cnn_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        cnn_history = train_model(cnn_model, train_loader, val_loader, cnn_criterion, cnn_optimizer, 
                                   device, "CNN", samples_per_class)
        cnn_test_acc = evaluate_on_test_set(cnn_model, test_loader, device)
        
        # Save CNN model
        cnn_path = get_overfit_sweep_path("cnn", samples_per_class)
        ensure_parent_dir(cnn_path)
        torch.save(cnn_model.state_dict(), cnn_path)
        
        results['cnn'][samples_per_class] = {
            'history': cnn_history,
            'test_acc': cnn_test_acc,
            'path': cnn_path
        }
        
        print(f"  CNN Test Accuracy: {cnn_test_acc:.2f}%")
        
        # Train ConvKAN
        kan_model = get_convkan_model(device, version="android")
        kan_criterion = nn.CrossEntropyLoss(weight=class_weights)
        kan_optimizer = torch.optim.AdamW(kan_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        kan_history = train_model(kan_model, train_loader, val_loader, kan_criterion, kan_optimizer,
                                  device, "ConvKAN", samples_per_class)
        kan_test_acc = evaluate_on_test_set(kan_model, test_loader, device)
        
        # Save ConvKAN model
        kan_path = get_overfit_sweep_path("convkan", samples_per_class)
        ensure_parent_dir(kan_path)
        torch.save(kan_model.state_dict(), kan_path)
        
        results['convkan'][samples_per_class] = {
            'history': kan_history,
            'test_acc': kan_test_acc,
            'path': kan_path
        }
        
        print(f"  ConvKAN Test Accuracy: {kan_test_acc:.2f}%")
    
    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    # Create output directory
    os.makedirs('overfitting_experiments', exist_ok=True)
    
    # Plot 1: Test accuracy vs sample size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cnn_test_accs = [results['cnn'][s]['test_acc'] for s in SAMPLE_SIZES]
    kan_test_accs = [results['convkan'][s]['test_acc'] for s in SAMPLE_SIZES]
    
    ax.plot(SAMPLE_SIZES, cnn_test_accs, marker='s', label='CNN', linewidth=2, markersize=8)
    ax.plot(SAMPLE_SIZES, kan_test_accs, marker='o', label='ConvKAN', linewidth=2, markersize=8)
    
    ax.set_xlabel('Training Samples per Class', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Overfitting Experiments: Test Accuracy vs Training Data Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels(SAMPLE_SIZES)
    
    plt.tight_layout()
    plt.savefig('overfitting_experiments/test_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: overfitting_experiments/test_accuracy_comparison.png")
    
    # Plot 2: Training curves for each sample size
    num_experiments = len(SAMPLE_SIZES)
    rows = (num_experiments + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    for idx, samples_per_class in enumerate(SAMPLE_SIZES):
        ax = axes[idx]
        
        cnn_hist = results['cnn'][samples_per_class]['history']
        kan_hist = results['convkan'][samples_per_class]['history']
        
        cnn_epochs = range(1, len(cnn_hist['val_accs']) + 1)
        kan_epochs = range(1, len(kan_hist['val_accs']) + 1)
        
        ax.plot(cnn_epochs, cnn_hist['val_accs'], label='CNN Val', linewidth=2)
        ax.plot(kan_epochs, kan_hist['val_accs'], label='ConvKAN Val', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title(f'{samples_per_class} samples/class', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(num_experiments, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Validation Accuracy Curves for Different Sample Sizes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('overfitting_experiments/validation_curves_all.png', dpi=150, bbox_inches='tight')
    print("Saved: overfitting_experiments/validation_curves_all.png")
    
    # Summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Samples/Class':<15} {'CNN Acc':<12} {'ConvKAN Acc':<12} {'Winner':<12} {'CNN Epochs':<12} {'KAN Epochs':<12}")
    print("-" * 85)
    
    for samples in SAMPLE_SIZES:
        cnn_acc = results['cnn'][samples]['test_acc']
        kan_acc = results['convkan'][samples]['test_acc']
        winner = "ConvKAN" if kan_acc > cnn_acc else ("CNN" if cnn_acc > kan_acc else "Tie")
        cnn_epochs = results['cnn'][samples]['history']['total_epochs']
        kan_epochs = results['convkan'][samples]['history']['total_epochs']
        
        print(f"{samples:<15} {cnn_acc:<12.2f} {kan_acc:<12.2f} {winner:<12} {cnn_epochs:<12} {kan_epochs:<12}")
    
    print("\n" + "="*80)
    print("OVERFITTING EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - overfitting_experiments/test_accuracy_comparison.png")
    print("  - overfitting_experiments/validation_curves_all.png")
    print("  - models/overfit_trials/cnn/malaria_cnn_overfit_*samples.pth")
    print("  - models/overfit_trials/convkan/malaria_convkan_overfit_*samples.pth")

if __name__ == "__main__":
    main()
