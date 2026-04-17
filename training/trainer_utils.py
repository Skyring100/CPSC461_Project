import gc
import json
import os
import psutil
import torch
import matplotlib.pyplot as plt

from utils import ensure_parent_dir

def make_history() -> dict:
    return {"train_loss": [], "val_acc": [], "epoch_times": []}


def get_initial_ram() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def get_initial_vram() -> float:
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cleanup(model, optimizer):
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_stats(results: dict, path: str):
    # numpy arrays are not JSON-serialisable; exclude them from the file.
    serialisable = {
        k: v for k, v in results.items()
        if k not in ("confusion_matrix", "all_preds", "all_labels")
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=4)
    print(f"Stats saved to {path}")


def save_training_plot(history: dict, model_name: str, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history["val_acc"], color="green", marker="o")
    ax1.set_title("Validation Accuracy (%)")
    ax2.plot(history["train_loss"], color="orange")
    ax2.set_title("Training Loss")
    fig.suptitle(f"{model_name} Training Metrics")
    ensure_parent_dir(path)
    plt.savefig(path)
    plt.close()
    print(f"Training plot saved to {path}")