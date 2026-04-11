import os
from torch.utils.data import DataLoader
from common import prepare_dataset_root, get_data_split
import medmnist
import matplotlib.pyplot as plt
import numpy as np


def get_malaria_dataloader():
    real_root = prepare_dataset_root()
    train_dataset, test_dataset = get_data_split(real_root)
    malaria_train_loader = DataLoader(train_dataset, batch_size=1)
    malaria_test_loader = DataLoader(test_dataset, batch_size=1)
    return malaria_train_loader, malaria_test_loader


def get_medmnist_dataloader(version = 'nodulemnist3d'):
    DataClass = getattr(medmnist, medmnist.INFO[version]['python_class'])
    DATASET_PATH = os.path.join(os.getcwd(), "medmnist", version)
    train_dataset = DataClass(split='train',  download=True, root=DATASET_PATH)
    test_dataset = DataClass(split='test',  download=True, root=DATASET_PATH)

    nodulemnist_train_loader = DataLoader(dataset=train_dataset, batch_size=1)
    nodulemnist_test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    return nodulemnist_train_loader, nodulemnist_test_loader

def count_labels(dataloader: DataLoader):
    labels = {0:0, 1:0}
    for image, l in dataloader:
        class_label = l[0].item()
        labels[class_label] += 1
    return labels


malaria_train, malaria_test = get_malaria_dataloader()
nodulemnist_train, nodulemnist_test = get_medmnist_dataloader()

malaria_train_count = count_labels(malaria_train)
malaria_test_count = count_labels(malaria_test)
nodulemnist_train_count = count_labels(nodulemnist_train)
nodulemnist_test_count = count_labels(nodulemnist_test)

print(malaria_train_count)
print(malaria_test_count)
print(nodulemnist_train_count)
print(nodulemnist_test_count)
'''
{0: 9890, 1: 9952}
{0: 2540, 1: 2421}
{0: 863, 1: 295}
{0: 246, 1: 64}
'''



