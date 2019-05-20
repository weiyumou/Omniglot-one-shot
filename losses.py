import math

import torch
import torch.nn as nn


class SlideLoss(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, anc_pos, anc_neg):
        diff = anc_pos - anc_neg
        diff[diff >= 0] = self.scale * diff[diff >= 0] / math.e + self.scale
        diff[diff < 0] = self.scale / torch.log(math.e - diff[diff < 0])
        return torch.mean(diff)

    def step(self, epoch_loss: float):
        self.scale = math.exp(self.scale / epoch_loss) * self.scale / epoch_loss
