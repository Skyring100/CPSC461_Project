import sys

from .models import get_model
from .visualizer import save_comparison_plot
from utils import get_comparison_chart_path
from .train_model import train_model
from .data import get_medmnist_loaders

VERSIONS = ["nodulemnist3d_light", "nodulemnist3d"]
TITLES = ["NoduleMNIST Light", "NoduleMNIST Bulky"]
MODELS = ["cnn", "convkan"]

def medmnist_loader_builder(batch_size):
    return get_medmnist_loaders(batch_size)

def medmnist_model_builder(device, version, model_name):
    return get_model(device, isConvKAN=(model_name == "convkan"), version=version, is3d=True)

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
            save_comparison_plot(version_stats, v, chart_path, TITLES[v_list.index(v)])


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) >= 2 else None
    m = sys.argv[2] if len(sys.argv) == 3 else None
    try:
        run_medmnist_training(v, m)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        sys.exit(0)