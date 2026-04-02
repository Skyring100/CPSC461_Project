from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import medmnist
import copy
from medmnist import INFO
from common import (MODELS_ROOT, get_model, count_parameters)
import os
import matplotlib.pyplot as plt
import sys
import time
import psutil
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Global dict to store results for both models
training_results = {}

def train_model(model_name: str, version: str):
    global training_results
    
    isConvKAN = True
    if (model_name.lower() == "convkan"):
        isConvKAN = True
    elif (model_name.lower() == "cnn"):
        isConvKAN = False
    else:
        print("Invalid model name, exiting")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*80)
    print(f"Training {model_name.upper()} Model for {version}")
    print("="*80)
    
    model = get_model(device, isConvKAN, version)
    num_params = count_parameters(model)
    print(f"\nModel Architecture:\n{model}\n")
    print(f"Total Parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    MODEL_DIRECTORY = os.path.join(MODELS_ROOT, "mednist")
    if(not os.path.exists(MODEL_DIRECTORY)): 
        os.makedirs(MODEL_DIRECTORY)
    MODEL_PATH = os.path.join(MODEL_DIRECTORY, f'medmnist_{model_name}_{version}.pth')
    DATASET_PATH = os.path.join(os.getcwd(), "medmnist", version)
    print(f'Dataset path: {DATASET_PATH}')
    os.makedirs(DATASET_PATH, exist_ok=True)

    MAX_EPOCHS = 100
    PATIENCE = 5
    BATCH_SIZE = 20

    DataClass = getattr(medmnist, INFO[version]['python_class'])

    # load the data
    print("\nDownloading dataset...")
    train_dataset = DataClass(split='train',  download=True, root=DATASET_PATH)
    test_dataset = DataClass(split='test',  download=True, root=DATASET_PATH)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Memory tracking
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    process = psutil.Process(os.getpid())
    peak_ram = 0
    peak_vram = 0
    
    print(f"\nStarting Training for {model_name.upper()} with version {version}")
    print("-"*80)
    
    best_acc, epochs_no_improve = 0.0, 0
    history = {"train_loss": [], "val_acc": [], "epoch_times": []}
    
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        for x, y in pbar:
            x, y = x.float().to(device), y.squeeze().long().to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update peak memory
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
            peak_ram = max(peak_ram, process.memory_info().rss / 1e9)
        
        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.float().to(device), y.squeeze().long().to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total
        history["val_acc"].append(acc)
        epoch_time = time.time() - epoch_start
        history["epoch_times"].append(epoch_time)
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}% | Time: {epoch_time:.2f}s")
        
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    total_time = time.time() - start_time
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_PATH)

    # Calculate final metrics with confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.float().to(device), y.squeeze().long().to(device)
            y_hat = model(x)
            _, predicted = torch.max(y_hat, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    
    avg_time_per_epoch = total_time / len(history["epoch_times"])
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS FOR {model_name.upper()}")
    print("="*80)
    print(f"Total Parameters: {num_params:,}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Peak Training RAM: {peak_ram:.2f} GB")
    print(f"Peak Training VRAM: {peak_vram:.2f} GB")
    print(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Avg Time per Epoch: {avg_time_per_epoch:.2f}s")
    print(f"Total Epochs Trained: {len(history['epoch_times'])}")
    print("="*80 + "\n")
    
    # Store results
    training_results[model_name.lower()] = {
        'num_params': num_params,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'peak_ram': peak_ram,
        'peak_vram': peak_vram,
        'total_time': total_time,
        'avg_epoch_time': avg_time_per_epoch,
        'num_epochs': len(history['epoch_times']),
        'train_loss': history['train_loss'],
        'val_acc': history['val_acc'],
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


if len(sys.argv) == 1:
    train_model("cnn", "nodulemnist3d")    
    train_model("convkan", "nodulemnist3d")
elif len(sys.argv) == 2:
    train_model(sys.argv[1], "nodulemnist3d")
else:
    train_model(sys.argv[1], sys.argv[2])

# Generate comprehensive comparison visualization
if len(training_results) == 2 and "cnn" in training_results and "convkan" in training_results:
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATION")
    print("="*80 + "\n")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
    
    # === STATS COMPARISON (TOP ROW) ===
    models_list = ['cnn', 'convkan']
    stats = training_results
    
    # 1. Parameters comparison
    ax1 = fig.add_subplot(gs[0, 0])
    params_vals = [stats[m]['num_params']/1e6 for m in models_list]
    colors = ['#3498db', '#e74c3c']
    bars1 = ax1.bar(models_list, params_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Parameters', fontsize=12, fontweight='bold')
    for bar, val in zip(bars1, params_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}M',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 1])
    acc_vals = [stats[m]['accuracy'] for m in models_list]
    bars2 = ax2.bar(models_list, acc_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    for bar, val in zip(bars2, acc_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. F1 Score comparison
    ax3 = fig.add_subplot(gs[0, 2])
    f1_vals = [stats[m]['f1'] for m in models_list]
    bars3 = ax3.bar(models_list, f1_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('F1 Score (Weighted)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.1])
    for bar, val in zip(bars3, f1_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Training time comparison
    ax4 = fig.add_subplot(gs[0, 3])
    time_vals = [stats[m]['total_time']/60 for m in models_list]
    bars4 = ax4.bar(models_list, time_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax4.set_title('Total Training Time', fontsize=12, fontweight='bold')
    for bar, val in zip(bars4, time_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}m',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # === METRICS TABLE (SECOND ROW) ===
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    table_data = []
    headers = ['Metric', 'CNN', 'ConvKAN']
    
    for model_name in models_list:
        row_data = {
            'params': f"{stats[model_name]['num_params']:,}",
            'accuracy': f"{stats[model_name]['accuracy']:.2f}%",
            'precision': f"{stats[model_name]['precision']:.4f}",
            'recall': f"{stats[model_name]['recall']:.4f}",
            'f1': f"{stats[model_name]['f1']:.4f}",
            'peak_ram': f"{stats[model_name]['peak_ram']:.2f} GB",
            'peak_vram': f"{stats[model_name]['peak_vram']:.2f} GB",
            'total_time': f"{stats[model_name]['total_time']/60:.2f} min",
            'avg_epoch': f"{stats[model_name]['avg_epoch_time']:.2f}s",
            'epochs': f"{stats[model_name]['num_epochs']}"
        }
        if model_name == 'cnn':
            cnn_row = row_data
        else:
            convkan_row = row_data
    
    table_data = [
        ['Total Parameters', cnn_row['params'], convkan_row['params']],
        ['Overall Accuracy', cnn_row['accuracy'], convkan_row['accuracy']],
        ['Precision (Weighted)', cnn_row['precision'], convkan_row['precision']],
        ['Recall (Weighted)', cnn_row['recall'], convkan_row['recall']],
        ['F1 Score (Weighted)', cnn_row['f1'], convkan_row['f1']],
        ['Peak Training RAM', cnn_row['peak_ram'], convkan_row['peak_ram']],
        ['Peak Training VRAM', cnn_row['peak_vram'], convkan_row['peak_vram']],
        ['Total Training Time', cnn_row['total_time'], convkan_row['total_time']],
        ['Avg Time per Epoch', cnn_row['avg_epoch'], convkan_row['avg_epoch']],
        ['Total Epochs', cnn_row['epochs'], convkan_row['epochs']]
    ]
    
    table = ax_table.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                          colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    # === CONFUSION MATRICES (THIRD ROW) ===
    for idx, model_name in enumerate(models_list):
        cm = stats[model_name]['confusion_matrix']
        ax_cm = fig.add_subplot(gs[2, idx])
        
        im = ax_cm.imshow(cm, cmap='Blues', aspect='auto')
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Negative', 'Positive'], fontsize=10)
        ax_cm.set_yticklabels(['Negative', 'Positive'], fontsize=10)
        ax_cm.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax_cm.set_ylabel('True', fontsize=11, fontweight='bold')
        ax_cm.set_title(f'Confusion Matrix - {model_name.upper()}', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] > cm.max()/2 else "black",
                                fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax_cm, label='Count')
    
    # === TRAINING CURVES (CONTINUATION) ===
    ax_loss = fig.add_subplot(gs[2, 2])
    for idx, model_name in enumerate(models_list):
        ax_loss.plot(stats[model_name]['train_loss'], label=f'{model_name.upper()} Loss',
                    color=colors[idx], linewidth=2, marker='o', markersize=3)
    ax_loss.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax_loss.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax_loss.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, alpha=0.3)
    
    ax_acc = fig.add_subplot(gs[2, 3])
    for idx, model_name in enumerate(models_list):
        ax_acc.plot(stats[model_name]['val_acc'], label=f'{model_name.upper()} Acc',
                   color=colors[idx], linewidth=2, marker='o', markersize=3)
    ax_acc.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax_acc.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax_acc.set_title('Validation Accuracy Curves', fontsize=12, fontweight='bold')
    ax_acc.legend(fontsize=10)
    ax_acc.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('NoduleMNIST Training Comparison: CNN vs ConvKAN', 
                fontsize=16, fontweight='bold', y=0.995)
    
    MODEL_DIRECTORY = os.path.join(MODELS_ROOT, "mednist")
    COMPARISON_PATH = os.path.join(MODEL_DIRECTORY, 'medmnist_comparison_nodulemnist3d.png')
    plt.savefig(COMPARISON_PATH, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to: {COMPARISON_PATH}\n")
    plt.close()
    
    print("="*80)
    print("TRAINING COMPLETE - All metrics and visualizations saved!")
    print("="*80)
