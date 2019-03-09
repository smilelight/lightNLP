import torch
import torch.nn as nn
from torchtext.vocab import Vectors

from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG
from ...base.model import BaseConfig, BaseModel


class LMConfig(BaseConfig):
    def __init__(self, word_vocab, vector_path, **kwargs):
        super(LMConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class RNNLM(BaseModel):
    def __init__(self, args):
        super(RNNLM, self).__init__(args)
        self.args = args
        self.hidden_dim = args.embedding_dim
        self.vocabulary_size = args.vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)

        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim,
                            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        # self.dropout = nn.Dropout(args.dropout)
        self.bath_norm = nn.BatchNorm1d(embedding_dimension).to(DEVICE)
        # self.bath_norm2 = nn.BatchNorm1d(vocabulary_size).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.vocabulary_size).to(DEVICE)

        # self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)

        return h0, c0
    
    def forward(self, sentence):
        x = self.embedding(sentence.to(DEVICE)).to(DEVICE)
        self.hidden = self.init_hidden(batch_size=sentence.size(1))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.view(-1, lstm_out.size(2))
        lstm_out = self.bath_norm(lstm_out)
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)
