"""
Unimodal image classification - transfer learn using some
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

res_mod = models.resnet18(pretrained=True)

print(res_mod)
