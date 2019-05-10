import torch.nn as nn
import math
import torch


class SlideLoss(nn.Module):
    def forward(self, anc, pos, neg):
        anc_pos = torch.dist(anc, pos)
        anc_neg = torch.dist(anc, neg)
        diff = anc_pos - anc_neg
        diff[diff >= 0] = diff[diff >= 0] / math.e + 1
        diff[diff < 0] = 1 / torch.log(math.e - diff[diff < 0])
        return torch.mean(diff)
