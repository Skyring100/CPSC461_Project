import kagglehub
import os
import shutil

# Define the name of the dataset folder
local_dataset_path = "malaria_dataset"

# Check if the dataset already exists
if os.path.exists(local_dataset_path):
    print(f"Dataset found locally at: {local_dataset_path}")
    path = local_dataset_path
else:
    print("Dataset not found locally. Downloading...")
    
    # Download latest version
    cached_path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
    
    # Move the files from the cache to your local project folder
    shutil.move(cached_path, local_dataset_path)
    
    path = local_dataset_path