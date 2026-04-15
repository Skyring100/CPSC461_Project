from common import get_convkan_model, CONVKAN_SAVE_PATH
from evaluation.evaluate_model import evaluate_model

model_func = get_convkan_model
save_path = CONVKAN_SAVE_PATH
cmap = "Blues"
print_missing = f"Error: No model weights found at {CONVKAN_SAVE_PATH}. Please run train_convkan.py first."
plot_title = "ConvKAN Confusion Matrix"

evaluate_model(model_func, save_path, cmap, print_missing, plot_title)