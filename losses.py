import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class SlideLoss(nn.Module):
    def forward(self, anc, pos, neg):
        anc_pos = F.pairwise_distance(anc, pos, p=2, keepdim=True)
        anc_neg = F.pairwise_distance(anc, neg, p=2, keepdim=True)
        diff = anc_pos - anc_neg
        diff[diff >= 0] = diff[diff >= 0] / math.e + 1
        diff[diff < 0] = 1 / torch.log(math.e - diff[diff < 0])
        return torch.mean(diff)
