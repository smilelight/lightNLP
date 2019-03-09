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


class MaLSTM(BaseModel):
    def __init__(self, args):
        super(MaLSTM, self).__init__(args)

        self.args = args
        self.hidden_dim = 300
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        self.pwd = torch.nn.PairwiseDistance(p=1)

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)

        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional,
                            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(DEVICE)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def forward(self, left, right):
        left_vec = self.embedding(left.to(DEVICE)).to(DEVICE)
        #         left_vec = pack_padded_sequence(left_vec, left_sent_lengths)
        right_vec = self.embedding(right.to(DEVICE)).to(DEVICE)
        #         right_vec = pack_padded_sequence(right_vec, right_sent_lengths)

        self.hidden = self.init_hidden(batch_size=left.size(1))

        left_lstm_out, (left_lstm_hidden, _) = self.lstm(left_vec, self.hidden)
        #         left_lstm_out, left_batch_size = pad_packed_sequence(left_lstm_out)
        #         assert torch.equal(left_sent_lengths, left_batch_size.to(DEVICE))

        right_lstm_out, (right_lstm_hidden, _) = self.lstm(right_vec, self.hidden)
        #         right_lstm_out, right_batch_size = pad_packed_sequence(right_lstm_out)
        #         assert torch.equal(right_sent_lengths, right_batch_size.to(DEVICE))

        return self.manhattan_distance(left_lstm_hidden[0], right_lstm_hidden[0])

    def manhattan_distance(self, left, right):
        return torch.exp(-self.pwd(left, right))
