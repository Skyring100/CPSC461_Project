import sys

from training.train_malaria import run_malaria_training
from training.train_medmnist import run_medmnist_training


def run_training(version=None, model_name=None):
    """ Runs everything:
        1. Malaria (Nano and Standard versions)
        2. MedMNIST (NoduleMNIST3D and Light versions)
    """

    MALARIA_VERSIONS = ["nano", "standard"]
    MEDMNIST_VERSIONS = ["nodulemnist3d", "nodulemnist3d_light"]

    print("Starting")
    
    # No args - run all
    if version is None:
        print("Running all models on all datasets")
        
        try:
            run_malaria_training()
        except Exception as e:
            print(f"Error during Malaria training: {e}")

        print("\n" + "="*20)

        try:
            run_medmnist_training()
        except Exception as e:
            print(f"Error during MedMNIST training: {e}")

    # Malaria versions
    elif version in MALARIA_VERSIONS:
        try:
            run_malaria_training(version, model_name)
        except Exception as e:
            print(f"Error during Malaria training: {e}")

    # Medmnist versions
    elif version in MEDMNIST_VERSIONS:
        try:
            run_medmnist_training(version, model_name)
        except Exception as e:
            print(f"Error during MedMNIST training: {e}")
    
    else:
        print(f"Error: Unknown version '{version}'.")
        print(f"Available: {MALARIA_VERSIONS + MEDMNIST_VERSIONS}")

    print("\nCompleted")



if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) >= 2 else None
    m = sys.argv[2] if len(sys.argv) >= 3 else None
    try:
        run_training(v, m)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(0)