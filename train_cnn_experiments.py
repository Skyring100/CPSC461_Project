# Import shared logic
from common import (
    get_cnn_model, CNN_EXPERIMENT_TEMPLATE
)

from train_model_experiments import train_model_experiments

train_model_experiments("CNN", get_cnn_model, CNN_EXPERIMENT_TEMPLATE)
