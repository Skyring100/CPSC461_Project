# Import shared logic
from common import get_cnn_model
import torch


from train_model_experiments import train_model_experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model_experiments("CNN", get_cnn_model(device, "nano"), device)
