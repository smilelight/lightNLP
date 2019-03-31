import torch
import torch.nn as nn
import torch.nn.functional as F

from ....base.model import BaseModel

from ..config import DEVICE


class SkipGramHierarchicalSoftmax(BaseModel):
    def __init__(self, args):
        super(SkipGramHierarchicalSoftmax, self).__init__(args)

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim

        self.word_embeddings = nn.Embedding(2*self.vocabulary_size-1, self.embedding_dimension,
                                            sparse=True).to(DEVICE)
        self.context_embeddings = nn.Embedding(2*self.vocabulary_size-1, self.embedding_dimension,
                                               sparse=True).to(DEVICE)

    def forward(self, pos_target, pos_path, neg_target, neg_path):
        pos_target_embedding = torch.sum(self.word_embeddings(pos_target), dim=1, keepdim=True)
        pos_path_embedding = self.context_embeddings(pos_path)
        pos_score = torch.bmm(pos_target_embedding, pos_path_embedding.transpose(2, 1)).squeeze()
        neg_target_embedding = torch.sum(self.word_embeddings(neg_target), dim=1, keepdim=True)
        neg_path_embedding = self.context_embeddings(neg_path)
        neg_score = torch.bmm(neg_target_embedding, neg_path_embedding.transpose(2, 1)).squeeze()
        pos_sigmoid_score = torch.lt(torch.sigmoid(pos_score), 0.5)
        neg_sigmoid_score = torch.gt(torch.sigmoid(neg_score), 0.5)
        sigmoid_score = torch.cat((pos_sigmoid_score, neg_sigmoid_score))
        sigmoid_score = torch.sum(sigmoid_score, dim=0).item() / sigmoid_score.size(0)
        return sigmoid_score

    def loss(self, pos_target, pos_path, neg_target, neg_path):
        pos_target_embedding = torch.sum(self.word_embeddings(pos_target), dim=1, keepdim=True)
        pos_path_embedding = self.context_embeddings(pos_path)
        pos_score = torch.bmm(pos_target_embedding, pos_path_embedding.transpose(2, 1)).squeeze()
        neg_target_embedding = torch.sum(self.word_embeddings(neg_target), dim=1, keepdim=True)
        neg_path_embedding = self.context_embeddings(neg_path)
        neg_score = torch.bmm(neg_target_embedding, neg_path_embedding.transpose(2, 1)).squeeze()
        pos_score = torch.sum(F.logsigmoid(-1 * pos_score))
        neg_score = torch.sum(F.logsigmoid(neg_score))
        loss = -1 * (pos_score + neg_score)
        return loss

