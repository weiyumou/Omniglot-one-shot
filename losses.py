import math

import torch
import torch.nn as nn


class SlideLoss(nn.Module):
    def forward(self, anc_pos, anc_neg):
        diff = anc_pos - anc_neg
        diff[diff >= 0] = diff[diff >= 0] / math.e + 1
        diff[diff < 0] = 1 / torch.log(math.e - diff[diff < 0])
        return torch.mean(diff)
