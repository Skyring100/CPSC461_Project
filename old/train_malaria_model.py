import copy
import os
import time
from typing import Callable

import numpy as np
import psutil
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from data import get_data_split, prepare_dataset_root
from .engine import train_one_epoch, validate
from .evaluator import print_final_report, run_final_evaluation
from models import count_parameters
from utils import ensure_parent_dir, get_model_path, get_stats_path, get_training_plot_path
from .trainer_utils import (
    reset_memory, cleanup, make_history, get_initial_ram, 
    get_initial_vram, save_stats, save_training_plot
)


def train_malaria_model(model_name: str, get_model_fn: Callable, version: str = "nano") -> dict:
    """Train one model (CNN or ConvKAN) on the malaria dataset.

    model_name: "cnn" or "convkan"
    get_model_fn: ``get_cnn_model`` or ``get_convkan_model`` from ``models.py``.
    version: Architecture size variant

    Returns a metrics dict.
    """
    reset_memory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = get_model_path(model_name, version)
    stats_path = get_stats_path(model_name, version)
    training_plots_path = get_training_plot_path(model_name, version)
    ensure_parent_dir(model_path)

    model = get_model_fn(device, version=version)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_loader, test_loader = _build_malaria_loaders()

    history, peak_ram, peak_vram = make_history(), get_initial_ram(), get_initial_vram()
    best_acc, epochs_no_improve = 0.0, 0
    best_model_wts = None

    process = psutil.Process(os.getpid())

    start_time = time.time()
    print(f"\nStarting {version.upper()} {model_name} Training...")

    for epoch in range(100):
        epoch_start = time.time()
        loss, p_ram, p_vram = train_one_epoch(model, train_loader, optimizer, criterion, device, process)

        peak_ram = max(peak_ram, p_ram)
        peak_vram = max(peak_vram, p_vram)

        acc = validate(model, test_loader, device)
        history["train_loss"].append(loss)
        history["val_acc"].append(acc)
        history["epoch_times"].append(time.time() - epoch_start)
        print(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                break

    total_time = time.time() - start_time
    model.load_state_dict(best_model_wts)

    results = run_final_evaluation(model, test_loader, device)
    results.update({
        "num_params":     count_parameters(model),
        "peak_ram":       peak_ram,
        "peak_vram":      peak_vram,
        "total_time":     total_time,
        "avg_epoch_time": float(np.mean(history["epoch_times"])),
        "num_epochs":     len(history["train_loss"]),
        "train_loss":     history["train_loss"],
        "val_acc":        history["val_acc"],
    })
    print_final_report(model_name, results)

    torch.save(model.state_dict(), model_path)
    save_stats(results, stats_path)
    save_training_plot(history, model_name, training_plots_path)

    cleanup(model, optimizer)
    return results


# ---------------------------------------------------------------------------
# Private helpers

def _build_malaria_loaders():
    real_root = prepare_dataset_root()
    train_dataset, test_dataset = get_data_split(real_root)
    return (
        DataLoader(train_dataset, batch_size=16, shuffle=True),
        DataLoader(test_dataset, batch_size=16, shuffle=False),
    )