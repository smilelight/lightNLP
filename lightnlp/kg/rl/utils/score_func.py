import torch

import torch.nn as nn

p1 = torch.nn.PairwiseDistance(p=1)
p2 = torch.nn.PairwiseDistance(p=2)


def l1_score(head, rel, tail):
    return p1(tail - head, rel)


def l2_score(head, rel, tail):
    return p2(tail - head, rel)
