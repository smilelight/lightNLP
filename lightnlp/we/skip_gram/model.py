import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base.model import BaseConfig, BaseModel

from .config import DEVICE, DEFAULT_CONFIG, Feature


class Config(BaseConfig):
    def __init__(self, word_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        for name, value in kwargs.items():
            setattr(self, name, value)


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


class CBOWHierarchicalSoftmax(BaseModel):
    def __init__(self, args):
        super(CBOWHierarchicalSoftmax, self).__init__(args)

        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim

        self.word_embeddings = nn.Embedding(2*self.vocabulary_size-1, self.embedding_dimension,
                                            sparse=True).to(DEVICE)
        self.context_embeddings = nn.Embedding(2*self.vocabulary_size-1, self.embedding_dimension,
                                               sparse=True).to(DEVICE)

    def forward(self, x):
        pass

    def loss(self, pos_context, pos_path, neg_context, neg_path):
        pass
