
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....base.model import BaseModel

from ..config import DEVICE


class CBOWNegativeSampling(BaseModel):
    def __init__(self, args):
        super(CBOWNegativeSampling, self).__init__(args)

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim

        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)
        self.context_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)

    def forward(self, context, target):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.context_embeddings(target)
        target_score = torch.bmm(target_embedding, context_embedding.unsqueeze(2))
        return torch.sigmoid(target_score)

    def loss(self, context, pos, neg):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        pos_embedding = self.context_embeddings(pos)
        neg_embedding = self.context_embeddings(neg).squeeze()
        pos_score = torch.bmm(pos_embedding, context_embedding.unsqueeze(2)).squeeze()
        neg_score = torch.bmm(neg_embedding, context_embedding.unsqueeze(2)).squeeze()
        pos_score = torch.sum(F.logsigmoid(pos_score), dim=0)
        neg_score = torch.sum(F.logsigmoid(-1 * neg_score), dim=0)
        return -1*(torch.sum(pos_score) + torch.sum(neg_score))


