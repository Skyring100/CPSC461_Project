import os
from torch.utils.data import DataLoader
from common import prepare_dataset_root, get_data_split
import medmnist


def get_malaria_dataloader():
    real_root = prepare_dataset_root()
    train_dataset, test_dataset = get_data_split(real_root)
    malaria_train_loader = DataLoader(train_dataset)
    malaria_test_loader = DataLoader(test_dataset)
    return malaria_train_loader, malaria_test_loader


def get_medmnist_dataloader(version = 'nodulemnist3d'):
    DataClass = getattr(medmnist, medmnist.INFO[version]['python_class'])
    DATASET_PATH = os.path.join(os.getcwd(), "medmnist", version)
    train_dataset = DataClass(split='train',  download=True, root=DATASET_PATH)
    test_dataset = DataClass(split='test',  download=True, root=DATASET_PATH)

    nodulemnist_train_loader = DataLoader(dataset=train_dataset)
    nodulemnist_test_loader = DataLoader(dataset=test_dataset)
    return nodulemnist_train_loader, nodulemnist_test_loader


malaria_train, malaria_test = get_malaria_dataloader()
nodulemnist_train, nodulemnist_test = get_malaria_dataloader()

