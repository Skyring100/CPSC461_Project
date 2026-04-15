import os
import json
import matplotlib.pyplot as plt

from config import MODELS_ROOT


def ensure_parent_dir(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def get_model_path(model_name: str, version: str) -> str:
    return os.path.join(MODELS_ROOT, version, f"{model_name}_{version}.pth")


def get_stats_path(model_name: str, version: str) -> str:
    return os.path.join(MODELS_ROOT, version, f"{model_name}_{version}_stats.json")


def get_comparison_chart_path(version: str) -> str:
    return os.path.join("comparison", version, f"cnn_convkan_{version}_comparison.png")


def get_training_plot_path(model_name: str, version: str) -> str:
    return os.path.join("training_plots", version, f"{model_name}_{version}_stats.png")


def _save_stats(results: dict, path: str):
    # numpy arrays are not JSON-serialisable; exclude them from the file.
    serialisable = {
        k: v for k, v in results.items()
        if k not in ("confusion_matrix", "all_preds", "all_labels")
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=4)
    print(f"Stats saved to {path}")


def _save_training_plot(history: dict, model_name: str, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history["val_acc"], color="green", marker="o")
    ax1.set_title("Validation Accuracy (%)")
    ax2.plot(history["train_loss"], color="orange")
    ax2.set_title("Training Loss")
    fig.suptitle(f"{model_name} Training Metrics")
    plt.savefig(path)
    plt.close()
    print(f"Training plot saved to {path}")