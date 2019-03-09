import torch
import torch.nn as nn
from torchtext.vocab import Vectors

from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG
from ...base.model import BaseConfig, BaseModel


class Config(BaseConfig):
    def __init__(self, word_vocab, label_vocab, vector_path, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.class_num = len(self.label_vocab)
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class SharedLSTM(BaseModel):
    def __init__(self, args):
        super(SharedLSTM, self).__init__(args)

        self.args = args
        self.hidden_dim = 300
        self.class_num = args.class_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)

        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional,
                            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        self.dropout_layer = nn.Dropout(self.dropout).to(DEVICE)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim * 2).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.class_num).to(DEVICE)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def forward(self, left, right):
        left_vec = self.embedding(left.to(DEVICE)).to(DEVICE)
        right_vec = self.embedding(right.to(DEVICE)).to(DEVICE)

        self.hidden = self.init_hidden(batch_size=left.size(1))

        left_lstm_out, (left_lstm_hidden, _) = self.lstm(left_vec, self.hidden)

        right_lstm_out, (right_lstm_hidden, _) = self.lstm(right_vec, self.hidden)

        merged = torch.cat((left_lstm_out[-1], right_lstm_out[-1]), dim=1)

        merged = self.dropout_layer(merged)

        merged = self.batch_norm(merged)

        predict = self.hidden2label(merged)

        return predict
