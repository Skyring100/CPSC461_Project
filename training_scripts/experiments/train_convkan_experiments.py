# Import shared logic
from common import (
    get_convkan_model, CONVKAN_EXPERIMENT_TEMPLATE
)

from train_model_experiments import train_model

train_model("CNN", get_convkan_model, CONVKAN_EXPERIMENT_TEMPLATE)
