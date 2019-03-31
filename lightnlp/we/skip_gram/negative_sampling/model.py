
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....base.model import BaseModel

from ..config import DEVICE


class SkipGramNegativeSampling(BaseModel):
    def __init__(self, args):
        super(SkipGramNegativeSampling, self).__init__(args)

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim

        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)
        self.context_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension).to(DEVICE)

    def forward(self, target, context):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.context_embeddings(context)
        target_score = torch.matmul(target_embedding, context_embedding.transpose(2, 1)).squeeze()
        return torch.sigmoid(target_score)

    def loss(self, target, pos, neg):
        target_embedding = self.word_embeddings(target)
        pos_embedding = self.context_embeddings(pos)
        neg_embedding = self.context_embeddings(neg).squeeze()
        pos_score = torch.matmul(target_embedding, pos_embedding.transpose(2, 1)).squeeze()
        neg_score = torch.matmul(target_embedding, neg_embedding.transpose(1, 2)).squeeze()
        pos_score = torch.sum(F.logsigmoid(pos_score), dim=0)
        neg_score = torch.sum(F.logsigmoid(-1 * neg_score), dim=0)
        return -1*(torch.sum(pos_score) + torch.sum(neg_score))


