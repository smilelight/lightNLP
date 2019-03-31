import torch
import torch.nn as nn
import torch.nn.functional as F

from ....base.model import BaseModel

from ..config import DEVICE


class SkipGramBase(BaseModel):
    def __init__(self, args):
        super(SkipGramBase, self).__init__(args)

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim

        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)
        self.linear = nn.Linear(self.embedding_dimension, self.vocabulary_size).to(DEVICE)

    def forward(self, target):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.linear(target_embedding).squeeze()
        return context_embedding

    def loss(self, target, context):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.linear(target_embedding).reshape(target_embedding.size(0), -1)
        return F.cross_entropy(context_embedding, context.view(-1))
