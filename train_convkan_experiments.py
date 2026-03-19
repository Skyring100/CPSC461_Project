# Import shared logic
from common import get_convkan_model
import torch


from train_model_experiments import train_model_experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model_experiments("CONVKAN", get_convkan_model(device, "nano"), device)
