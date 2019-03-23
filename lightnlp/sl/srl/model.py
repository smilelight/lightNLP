import torch
import torch.nn as nn
from torchcrf import CRF
from torchtext.vocab import Vectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG
from ...base.model import BaseConfig, BaseModel


class Config(BaseConfig):
    def __init__(self, word_vocab, pos_vocab, tag_vocab, vector_path, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.tag_vocab = tag_vocab
        self.tag_num = len(self.tag_vocab)
        self.vocabulary_size = len(self.word_vocab)
        self.pos_size = len(self.pos_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class BiLstmCrf(BaseModel):
    def __init__(self, args):
        super(BiLstmCrf, self).__init__(args)
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
        pos_size = args.pos_size
        pos_dim = args.pos_dim

        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.word_embedding = nn.Embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)
        self.pos_embedding = nn.Embedding(pos_size, pos_dim).to(DEVICE)

        self.lstm = nn.LSTM(embedding_dimension + pos_dim + 1, self.hidden_dim // 2, bidirectional=self.bidirectional,
                            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(DEVICE)
        self.crflayer = CRF(self.tag_num).to(DEVICE)

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

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def loss(self, x, sent_lengths, pos, rel, y):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, pos, rel, sent_lengths)
        return self.crflayer(emissions, y, mask=mask)

    def forward(self, x, poses, rels, sent_lengths):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, poses, rels, sent_lengths)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, poses, rels, sent_lengths):
        word = self.word_embedding(sentence.to(DEVICE)).to(DEVICE)
        pos = self.pos_embedding(poses.to(DEVICE)).to(DEVICE)
        rels = rels.view(rels.size(0), rels.size(1), 1).float().to(DEVICE)
        x = torch.cat((word, pos, rels), dim=2)
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(DEVICE))
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)
