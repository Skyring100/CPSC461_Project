# Import shared logic
from common import (
    get_cnn_model, CNN_EXPERIMENT_TEMPLATE
)

from train_model_experiments import train_model

train_model("CNN", get_cnn_model, CNN_EXPERIMENT_TEMPLATE)
