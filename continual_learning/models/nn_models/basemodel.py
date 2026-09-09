import torch
from torch import nn

__all__ = ['BaseModel']


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x, labels=None, experience_id=None):
        raise NotImplementedError

    def forward_all_layers(self, x, experience_id=None):
        raise NotImplementedError

