"""
Joiner modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .position_encoding import build_position_encoding

class Joiner(nn.Sequential):
    def __init__(self,position_embedding):
        super().__init__(position_embedding)

    def forward(self, x, mask):
        pos = self[0](x, mask)
        return x, pos


def build_joiner(args):
    position_embedding = build_position_encoding(args)
    model = Joiner(position_embedding)
    return model
