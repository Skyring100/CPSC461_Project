from data import get_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_model_info(isConvKAN: bool, version: str):
    m = get_model(device, isConvKAN, version)
    dataset_name = ("Nodule" if version.find("nodulemnist3d") != -1 else "Malaria")
    density_name = ("Light" if version=="nano" or version=="nodulemnist3d_light" else "Bulky")
    base_model_name = ("ConvKAN" if isConvKAN else "CNN")
    name =  f"{density_name}_{dataset_name}_{base_model_name}"
    parameter_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"---------------------------------------{name}---------------------------------------")
    print(f"Number of params:{parameter_count}")
    print(m)
    print(f"----------------------------------------------------------------------------------------------")

    return m


light_cnn_malaria = print_model_info(False, "nano")
light_convkan_malaria = print_model_info(True, "nano")

bulky_cnn_malaria = print_model_info(False, "standard")
bulky_convkan_malaria = print_model_info(True, "standard")

light_cnn_nodule = print_model_info(False, "nodulemnist3d_light")
light_convkan_nodule = print_model_info(True, "nodulemnist3d_light")

bulky_cnn_nodule = print_model_info(False, "nodulemnist3d")
bulky_convkan_nodule = print_model_info(True, "nodulemnist3d")