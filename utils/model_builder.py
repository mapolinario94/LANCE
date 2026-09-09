import argparse
from pathlib import Path
import time
from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

__all__ = ["build_model_and_size", "freeze_all", "unfreeze_last_n_convs", "get_parent_and_child_name", "unfreeze_last_linear"]

def _maybe_build_mcunet(net_id, pretrained=True):
    try:
        from mcunet.model_zoo import build_model
        return build_model(net_id=net_id, pretrained=pretrained)  # -> (model, image_size, desc)
    except Exception as e:
        raise RuntimeError(
            "MCUNet is not installed or not importable. "
            "Install with `pip install mcunet`, and ensure the chosen net_id exists."
        ) from e

# -----------------------
# Model builders & heads
# -----------------------
def build_model_and_size(arch: str, num_classes: int, net_id: str = "mcunet-in1", pretrain_model_path=None):
    """
    Returns (model, image_size)
    """
    arch = arch.lower()
    if arch == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        _replace_mobilenetv2_head(model, num_classes)
        img_size = 224

    elif arch == "resnet18":
        if pretrain_model_path is None:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet18(weights=weights)
        _replace_resnet_head(model, num_classes)
        img_size = 224
        if pretrain_model_path:
            model.load_state_dict(torch.load(pretrain_model_path)["state_dict"])

    elif arch == "resnet34":
        if pretrain_model_path is None:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet34(weights=weights)
        _replace_resnet_head(model, num_classes)
        img_size = 224
        if pretrain_model_path:
            model.load_state_dict(torch.load(pretrain_model_path)["state_dict"])

    elif arch == "mcunet":
        model, img_size, _desc = _maybe_build_mcunet(net_id=net_id, pretrained=True)
        _replace_last_linear(model, num_classes)

    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model, img_size


def _replace_mobilenetv2_head(model: nn.Module, num_classes: int):
    in_features = model.classifier[-1].in_features
    new_classifier = nn.Sequential(*[m for m in model.classifier[:-1]],
                                   nn.Linear(in_features, num_classes))
    model.classifier = new_classifier


def _replace_resnet_head(model: nn.Module, num_classes: int):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)


def _replace_last_linear(model: nn.Module, num_classes: int):
    """
    Generic fallback: find the *last* nn.Linear in the module tree and replace it.
    Works for MCUNet and most custom heads.
    """
    last_qn = None
    for qn, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_qn = qn
    if last_qn is None:
        raise RuntimeError("Could not find any nn.Linear to replace.")
    parent, child = get_parent_and_child_name(model, last_qn)
    old = getattr(parent, child)
    new = nn.Linear(old.in_features, num_classes, bias=(old.bias is not None))
    new.to(device=old.weight.device, dtype=old.weight.dtype)
    setattr(parent, child, new)

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_n_convs(model: nn.Module, n: int, include_classifier: bool = True):
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return
    n = max(0, min(n, len(convs)))
    for m in convs[-n:]:
        for p in m.parameters():
            p.requires_grad = True
    if include_classifier:
        unfreeze_last_linear(model)


def unfreeze_last_linear(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        for p in last.parameters():
            p.requires_grad = True


def get_parent_and_child_name(model: nn.Module, qualified_name: str):
    if "." in qualified_name:
        parent_name, child_name = qualified_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent, child_name = model, qualified_name
    return parent, child_name