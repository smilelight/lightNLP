import torch
import torch.nn as nn
import torch.nn.functional as F

p1 = torch.nn.PairwiseDistance(p=1)
p2 = torch.nn.PairwiseDistance(p=2)


def l1_score(vec1, vec2):
    return p1(vec1, vec2)


def l2_score(vec1, vec2):
    return p2(vec1, vec2)


def cos_score(vec1, vec2):
    return F.cosine_similarity(vec1, vec2)
