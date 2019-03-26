import torch.nn as nn
import torch.nn.functional as F

from .....base.model import BaseConfig, BaseModel
from ...utils.score_func import l1_score, l2_score
from .config import DEVICE, DEFAULT_CONFIG


class Config(BaseConfig):
    def __init__(self, entity_vocab, rel_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.entity_vocab = entity_vocab
        self.rel_vocab = rel_vocab
        self.entity_num = len(self.entity_vocab)
        self.rel_num = len(self.rel_vocab)
        for name, value in kwargs.items():
            setattr(self, name, value)


class TransE(BaseModel):
    def __init__(self, args):
        super(TransE, self).__init__(args)

        self.entity_num = args.entity_num
        self.rel_num = args.rel_num
        self.embedding_dimension = args.embedding_dim
        self.entity_embedding = nn.Embedding(self.entity_num, self.embedding_dimension).to(DEVICE)
        self.rel_embedding = nn.Embedding(self.rel_num, self.embedding_dimension).to(DEVICE)
        if args.score_func == 'l1':
            self.score_func = l1_score
        else:
            self.score_func = l2_score

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embedding.weight)
        nn.init.xavier_normal_(self.rel_embedding.weight)

    def forward(self, head, rel, tail):
        batch_size = head.size(0)
        vec_head = self.entity_embedding(head).view(-1, self.embedding_dimension)
        vec_rel = self.rel_embedding(rel).view(-1, self.embedding_dimension)
        vec_tail = self.entity_embedding(tail).view(-1, self.embedding_dimension)

        vec_head = F.normalize(vec_head)
        vec_rel = F.normalize(vec_rel)
        vec_tail = F.normalize(vec_tail)

        return self.score_func(vec_head, vec_rel, vec_tail)
        # return self.score_func(vec_head, vec_rel, vec_tail) / batch_size
        # return self.score_func(vec_head, vec_rel, vec_tail) / self.embedding_dimension

