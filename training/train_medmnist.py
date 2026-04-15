import sys
import os
import medmnist 

from medmnist import INFO
from models import get_model
from .visualizer import save_comparison_plot
from utils import get_comparison_chart_path
from .train_model import train_model
from torch.utils.data import DataLoader

VERSIONS = ["nodulemnist3d", "nodulemnist3d_light"]
MODELS = ["cnn", "convkan"]

def medmnist_loader_builder(version, batch_size):
    DataClass = getattr(medmnist, INFO["nodulemnist3d"]["python_class"])
    root = os.path.join(os.getcwd(), "medmnist", version)
    return DataLoader(DataClass(split="train", download=True, root=root), batch_size=batch_size, shuffle=True), \
           DataLoader(DataClass(split="test", download=True, root=root), batch_size=batch_size, shuffle=False)

def medmnist_model_builder(device, version, model_name):
    return get_model(device, isConvKAN=(model_name == "convkan"), version=version)

def run_medmnist_training(version, model_name):
    """Run MedMNIST training for the given model/version (or all combinations)"""
    v_list = [version] if version else VERSIONS
    m_list = [model_name] if model_name else MODELS

    for v in v_list:
        version_stats = {}
        for m in m_list:
            version_stats[m] = train_model(m, v, medmnist_model_builder, medmnist_loader_builder, batch_size=20)

        if "cnn" in version_stats and "convkan" in version_stats:
            chart_path = get_comparison_chart_path(v)
            save_comparison_plot(version_stats, v, chart_path)


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) >= 2 else None
    m = sys.argv[2] if len(sys.argv) == 3 else None
    try:
        run_medmnist_training(v, m)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        sys.exit(0)