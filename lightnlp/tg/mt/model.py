from .config import DEVICE, DEFAULT_CONFIG
from .models.encoder import Encoder
from .models.decoder import Decoder
from .models.seq2seq import Seq2Seq
from ...base.model import BaseConfig, BaseModel


class MTConfig(BaseConfig):
    def __init__(self, source_word_vocab, target_word_vocab, source_vector_path, target_vector_path, **kwargs):
        super(MTConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.source_word_vocab = source_word_vocab
        self.source_vocabulary_size = len(self.source_word_vocab)
        self.source_vector_path = source_vector_path
        self.target_word_vocab = target_word_vocab
        self.target_vocabulary_size = len(self.target_word_vocab)
        self.target_vector_path = target_vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class MTSeq2Seq(BaseModel):
    def __init__(self, args):
        super(MTSeq2Seq, self).__init__(args)
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.source_embedding_dim = args.source_embedding_dim
        self.target_embedding_dim = args.target_embedding_dim
        self.source_vector_path = args.source_vector_path
        self.target_vector_path = args.target_vector_path
        self.source_vocabulary_size = args.source_vocabulary_size
        self.target_vocabulary_size = args.target_vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        encoder = Encoder(self.source_vocabulary_size, self.source_embedding_dim, self.hidden_dim,
                          self.num_layers, self.dropout).to(DEVICE)
        decoder = Decoder(self.hidden_dim, self.target_embedding_dim, self.target_vocabulary_size,
                          self.num_layers, self.dropout,
                          args.method).to(DEVICE)
        self.seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)
