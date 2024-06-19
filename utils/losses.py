import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


def kl_divergence(p, q, eps=1e-8):
    """KL divergence"""
    p = p + eps
    q = q + eps
    return torch.sum(p * torch.log(p / q), dim=1)


def js_divergence(p, q, eps=1e-8):
    """JS divergence"""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


class FuzzyClassificationLoss(nn.Module):
    """Fuzzy classification loss"""

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean', softmax: bool = False):
        super(FuzzyClassificationLoss, self).__init__()
        if weight is not None and weight.min() < 0:
            raise ValueError('"weight" should greater than or equal to 0.')
        self.weight = weight.unsqueeze(-1).unsqueeze(-1) if weight is not None else weight
        self.reduction = reduction
        self.softmax = softmax

    def forward(self, input: Tensor) -> Tensor:
        input = F.softmax(input, dim=1)
        if self.weight is None:
            pixel_loss = input[:, 0, :, :] * input[:, 1, :, :]
        else:
            pixel_loss = -(1 - torch.prod((1-input).pow(self.weight), dim=1)).log()
        if self.reduction == 'mean':
            loss = pixel_loss.mean(dim=(0, 1, 2))
        elif self.reduction == 'sum':
            loss = pixel_loss.mean(dim=(1, 2)).sum(dim=0)
        else:
            loss = pixel_loss.mean(dim=(1, 2))
        return loss

    def get_fuzzy_region(self, input):
        input_prob = F.softmax(input, dim=1)
        padded_output = F.pad(input_prob, pad=(1, 1, 1, 1), mode='constant', value=0)

        batch_size = len(input)
        grid_size = 7  # grid patch size
        fuzzy_regions = torch.empty(batch_size, 2, 3, 3)

        for i in range(batch_size):
            class1_probabilities = input[i, 1, :, :]
            _, max_index = torch.max(class1_probabilities.view(-1), 0)
            max_y, max_x = max_index // grid_size, max_index % grid_size
            max_y_padded, max_x_padded = max_y + 1, max_x + 1
            region = padded_output[i, :, max_y_padded-1:max_y_padded+2, max_x_padded-1:max_x_padded+2]
            fuzzy_regions[i] = region

        return fuzzy_regions
