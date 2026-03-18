# Import shared logic
from common import (
    get_convkan_model, CONVKAN_OVERFITTING_PATH
)

from train_model_overfitting import train_model

train_model("CNN", get_convkan_model, CONVKAN_OVERFITTING_PATH)
