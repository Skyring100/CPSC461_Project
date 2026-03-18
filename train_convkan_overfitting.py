# Import shared logic
from common import (
    get_convkan_model, CONVKAN_OVERFITTING_PATH
)

from train_model_overfitting import train_model_overfitting

train_model_overfitting("CNN", get_convkan_model, CONVKAN_OVERFITTING_PATH)
