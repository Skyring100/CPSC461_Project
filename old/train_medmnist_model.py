import copy
import os
import time

import medmnist
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.utils.data as data

from medmnist import INFO
from .engine import train_one_epoch, validate
from .evaluator import print_final_report, run_final_evaluation
from models import count_parameters, get_model
from utils import ensure_parent_dir, get_model_path, get_stats_path, get_training_plot_path
from .trainer_utils import (
    reset_memory, cleanup, make_history, get_initial_ram, 
    get_initial_vram, save_stats, save_training_plot
)


def train_medmnist_model(model_name: str, version: str) -> dict:
    """Train one model (CNN or ConvKAN) on a MedMNIST dataset version.

    model_name: "cnn" or "convkan"
    version: Medmnist model version

    Returns a metrics dict.
    """
    reset_memory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = get_model_path(model_name, version)
    stats_path = get_stats_path(model_name, version)
    training_plots_path = get_training_plot_path(model_name, version)
    ensure_parent_dir(model_path)

    model = get_model(device, isConvKAN=(model_name.lower() == "convkan"), version=version)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = _build_medmnist_loaders(version)

    history, peak_ram, peak_vram = make_history(), get_initial_ram(), get_initial_vram()
    best_acc, epochs_no_improve = 0.0, 0
    best_model_wts = None

    process = psutil.Process(os.getpid())

    start_time = time.time()
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

    model.load_state_dict(best_model_wts)
    results = run_final_evaluation(model, test_loader, device)
    results.update({
        "num_params":     count_parameters(model),
        "peak_ram":       peak_ram,
        "peak_vram":      peak_vram,
        "total_time":     time.time() - start_time,
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


def _build_medmnist_loaders(version: str):
    DataClass = getattr(medmnist, INFO["nodulemnist3d"]["python_class"])
    dataset_path = os.path.join(os.getcwd(), "medmnist", version)
    train_loader = data.DataLoader(
        DataClass(split="train", download=True, root=dataset_path),
        batch_size=20, shuffle=True,
    )
    test_loader = data.DataLoader(
        DataClass(split="test", download=True, root=dataset_path),
        batch_size=20, shuffle=False,
    )
    return train_loader, test_loader