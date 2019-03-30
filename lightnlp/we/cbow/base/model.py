import torch
import torch.nn as nn
import torch.nn.functional as F

from ....base.model import BaseModel

from ..config import DEVICE


class CBOWBase(BaseModel):
    def __init__(self, args):
        super(CBOWBase, self).__init__(args)

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim

        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)
        self.linear = nn.Linear(self.embedding_dimension, self.vocabulary_size).to(DEVICE)

    def forward(self, context):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.linear(context_embedding)
        return target_embedding

    def loss(self, context, target):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.linear(context_embedding)
        return F.cross_entropy(target_embedding, target.view(-1))
