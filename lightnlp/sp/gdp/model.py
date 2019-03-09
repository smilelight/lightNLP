import torch
import torch.nn as nn
from torchcrf import CRF
from torchtext.vocab import Vectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG
from ...base.model import BaseConfig, BaseModel
from .components.dropout import IndependentDropout, SharedDropout
from .components import LSTM, MLP, Biaffine


class Config(BaseConfig):
    def __init__(self, word_vocab, pos_vocab, ref_vocab, vector_path, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.ref_vocab = ref_vocab
        self.pos_num = len(self.pos_vocab)
        self.ref_num = len(self.ref_vocab)
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class BiaffineParser(BaseModel):
    def __init__(self, args):
        super(BiaffineParser, self).__init__(args)
        self.args = args
        self.hidden_dim = args.lstm_hidden
        # self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        # self.num_layers = args.num_layers
        self.lstm_layters = args.lstm_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path

        vocabulary_size = args.vocabulary_size
        word_dim = args.word_dim
        pos_num = args.pos_num
        pos_dim = args.pos_dim

        # the embedding layer
        self.word_embedding = nn.Embedding(vocabulary_size, word_dim).to(DEVICE)
        vectors = Vectors(args.vector_path).vectors
        self.pretrained_embedding = nn.Embedding.from_pretrained(vectors).to(DEVICE)
        self.pos_embedding = nn.Embedding(pos_num, pos_dim).to(DEVICE)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout).to(DEVICE)

        # if args.static:
        #     logger.info('logging word vectors from {}'.format(args.vector_path))
        #     vectors = Vectors(args.vector_path).vectors
        #     self.word_embedding = nn.Embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)

        # the word-lstm layer
        self.lstm = LSTM(word_dim + pos_dim, self.hidden_dim, bidirectional=self.bidirectional,
                            num_layers=self.lstm_layters, dropout=args.lstm_dropout).to(DEVICE)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout).to(DEVICE)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_arc, dropout=args.mlp_dropout).to(DEVICE)
        self.mlp_arc_d = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_arc, dropout=args.mlp_dropout).to(DEVICE)
        self.mlp_rel_h = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_rel, dropout=args.mlp_dropout).to(DEVICE)
        self.mlp_rel_d = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_rel, dropout=args.mlp_dropout).to(DEVICE)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.mlp_arc, bias_x=True, bias_y=False).to(DEVICE)
        self.rel_attn = Biaffine(n_in=args.mlp_rel, n_out=args.ref_num, bias_x=True, bias_y=True).to(DEVICE)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embedding.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def forward(self, words, tags):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        embed = self.pretrained_embedding(words)
        embed += self.word_embedding(words.masked_fill_(words.ge(self.word_embedding.num_embeddings), 0))
        tag_embed = self.pos_embedding(tags)
        embed, tag_embed = self.embed_dropout(embed, tag_embed)
        # concatenate the word and tag representations
        x = torch.cat((embed, tag_embed), dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the LSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_((1 - mask).unsqueeze(1), float('-inf'))

        return s_arc, s_rel
