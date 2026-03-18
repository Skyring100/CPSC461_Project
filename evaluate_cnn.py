from common import get_cnn_model, CNN_SAVE_PATH
from evaluate_model import evaluate_model

model_func = get_cnn_model
save_path = CNN_SAVE_PATH
cmap = "Reds"
print_missing = f"Error: No CNN weights found at {CNN_SAVE_PATH}. Please run train_cnn.py first."
plot_title = "CNN Confusion Matrix"

evaluate_model(model_func, save_path, cmap, print_missing, plot_title)