import torch
import torch.nn as nn
from convkan import ConvKAN, LayerNorm2D
from ConvKAN3D.ConvKAN3D import effConvKAN3D

from config import IMG_SIZE


# ---------------------------------------------------------------------------
# Block builders

def _add_2d_block(model: nn.Sequential, isConvKAN: bool, in_ch: int, out_ch: int, version: str = "standard"):
    """Append a 2-D convolutional block to a model."""
    if isConvKAN:
        g, s = {
            "nano":    (2, 1),
            "pico":    (2, 2),
            "android": (3, 2),
        }.get(version, (5, 3))
        model.append(ConvKAN(in_ch, out_ch, kernel_size=3, stride=1, padding=1, grid_size=g, spline_order=s))
        model.append(LayerNorm2D(out_ch))
    else:
        model.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        model.append(nn.BatchNorm2d(out_ch))
        model.append(nn.LeakyReLU())


def _add_3d_block(model: nn.Sequential, isConvKAN: bool, in_ch: int, out_ch: int, version: str = "standard"):
    """Append a 3-D convolutional block to a model."""
    if isConvKAN:
        g = 6 if version == "nodulemnist3d" else 5
        model.append(effConvKAN3D(in_ch, out_ch, kernel_size=3, stride=1, padding=1, grid_size=g, spline_order=3))
    else:
        model.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        model.append(nn.LeakyReLU())


# ---------------------------------------------------------------------------
# Channel configs per version

def _get_channels(version: str, isConvKAN: bool) -> list[int]:
    mapping = {
        "nano":               [3, 2, 4, 4, 4]      if isConvKAN else [3, 4, 8, 10, 12],
        "pico":               [3, 4, 8, 12, 16],
        "android":            [3, 8, 16, 16, 32],
        "simple":             [3, 16, 32, 32, 64],
        "nodulemnist3d":      [1, 16, 32, 64, 128] if isConvKAN else [1, 64, 128, 256, 512],
        "nodulemnist3d_light":[1, 4, 4, 16, 16]    if isConvKAN else [1, 16, 32, 32, 64],
    }
    return mapping.get(version, [3, 32, 64, 128, 256])  # standard


# ---------------------------------------------------------------------------
# Backbone (shared between 2-D and 3-D)

def _build_backbone(model: nn.Sequential, channels: list[int], isConvKAN: bool, version: str, is3d: bool):
    current_in = channels[0]
    for out_ch in channels[1:]:
        if is3d:
            _add_3d_block(model, isConvKAN, current_in, out_ch, version)
            model.append(nn.MaxPool3d(2))
        else:
            _add_2d_block(model, isConvKAN, current_in, out_ch, version)
            model.append(nn.MaxPool2d(2))
        current_in = out_ch


# ---------------------------------------------------------------------------
# Classification heads

def _add_2d_head(model: nn.Sequential, channels: list[int], isConvKAN: bool, version: str):
    if version in ("android", "pico", "nano"):
        if isConvKAN:
            grid = 2 if version in ("pico", "nano") else 3
            model.append(ConvKAN(channels[-1], 2, kernel_size=1, grid_size=grid))
            model.append(nn.AdaptiveAvgPool2d(1))
            model.append(nn.Flatten())
        else:
            model.append(nn.AdaptiveAvgPool2d(1))
            model.append(nn.Flatten())
            model.append(nn.Linear(channels[-1], 2))
    else: # standard / simple
        if isConvKAN:
            model.append(nn.AdaptiveAvgPool2d(1))
            model.append(ConvKAN(channels[-1], 2, kernel_size=1))
            model.append(nn.Flatten())
        else:
            model.append(nn.Flatten())
            # Probe the in-progress model to get the flattened size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, channels[0], IMG_SIZE, IMG_SIZE)
                flatten_size = model(dummy).shape[1]
                print(flatten_size)
            if version == "simple":
                model.append(nn.Linear(flatten_size, 2))
            else:  # standard / default
                model.append(nn.Linear(flatten_size, 256))
                model.append(nn.LeakyReLU())
                model.append(nn.Linear(256, 2))


def _add_3d_head(model: nn.Sequential, channels: list[int], isConvKAN: bool, version: str):
    if version == "nodulemnist3d":
        if isConvKAN:
            model.append(effConvKAN3D(channels[-1], 128, kernel_size=1, grid_size=6, spline_order=3))
            model.append(nn.Flatten())
            model.append(nn.Linear(128, 64))
            model.append(nn.LeakyReLU())
            model.append(nn.Linear(64, 2))
        else:
            model.append(nn.AdaptiveAvgPool3d(1))
            model.append(nn.Flatten())
            model.append(nn.Linear(channels[-1], 256))
            model.append(nn.LeakyReLU())
            model.append(nn.Linear(256, 128))
            model.append(nn.LeakyReLU())
            model.append(nn.Linear(128, 2))
    else:  # nodulemnist3d_light
        if isConvKAN:
            model.append(effConvKAN3D(channels[-1], 2, kernel_size=1, grid_size=2))
            model.append(nn.AdaptiveAvgPool3d(1))
            model.append(nn.Flatten())
        else:
            model.append(nn.AdaptiveAvgPool3d(1))
            model.append(nn.Flatten())
            model.append(nn.Linear(channels[-1], 2))


# ---------------------------------------------------------------------------
# Public functions

def get_model(device, isConvKAN: bool, version: str = "standard") -> nn.Sequential:
    is3d = version in ("nodulemnist3d", "nodulemnist3d_light")
    channels = _get_channels(version, isConvKAN)

    model = nn.Sequential()
    _build_backbone(model, channels, isConvKAN, version, is3d)

    if is3d:
        _add_3d_head(model, channels, isConvKAN, version)
    else:
        _add_2d_head(model, channels, isConvKAN, version)

    return model.to(device)


def get_convkan_model(device, version: str = "standard") -> nn.Sequential:
    return get_model(device, isConvKAN=True, version=version)


def get_cnn_model(device, version: str = "standard") -> nn.Sequential:
    return get_model(device, isConvKAN=False, version=version)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)