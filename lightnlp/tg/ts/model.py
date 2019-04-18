from .config import DEVICE, DEFAULT_CONFIG
from .models.encoder import Encoder
from .models.decoder import Decoder
from .models.seq2seq import Seq2Seq
from ...base.model import BaseConfig, BaseModel


class TSConfig(BaseConfig):
    def __init__(self, word_vocab, vector_path, **kwargs):
        super(TSConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class TSSeq2Seq(BaseModel):
    def __init__(self, args):
        super(TSSeq2Seq, self).__init__(args)
        self.args = args
        self.hidden_dim = args.embedding_dim
        self.vocabulary_size = args.vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        encoder = Encoder(vocabulary_size, embedding_dimension, self.hidden_dim, self.num_layers,
                          self.dropout).to(DEVICE)
        decoder = Decoder(self.hidden_dim, embedding_dimension, vocabulary_size, self.num_layers, self.dropout,
                          args.method).to(DEVICE)
        self.seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)
