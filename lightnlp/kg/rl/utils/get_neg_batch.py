import random

import torch

random.seed(2019)


def get_neg_batch(head, tail, entity_num):
    neg_head = head.clone()
    neg_tail = tail.clone()
    if random.random() > 0.5:
        offset_tensor = torch.randint_like(neg_head, entity_num)
        neg_head = (neg_head + offset_tensor) % entity_num
    else:
        offset_tensor = torch.randint_like(neg_tail, entity_num)
        neg_tail = (neg_tail + offset_tensor) % entity_num
    return neg_head, neg_tail




