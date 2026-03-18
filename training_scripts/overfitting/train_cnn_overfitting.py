# Import shared logic
from common import (
    get_cnn_model, CNN_OVERFITTING_PATH
)

from train_model_overfitting import train_model

train_model("CNN", get_cnn_model, CNN_OVERFITTING_PATH)
