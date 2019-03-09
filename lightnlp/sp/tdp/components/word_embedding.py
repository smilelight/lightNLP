import torch
import torch.nn as nn
from torchtext.vocab import Vectors

from ..config import DEVICE
from ....utils.log import logger


class VanillaWordEmbeddingLookup(nn.Module):
    """
    A component that simply returns a list of the word embeddings as
    autograd Variables.
    """

    def __init__(self, vocabulary_size, embedding_dim, vector_path=None, non_static=False):
        super(VanillaWordEmbeddingLookup, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        self.output_dim = embedding_dim

        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dim).to(DEVICE)
        if vector_path:
            logger.info('logging word vectors from {}'.format(vector_path))
            word_vectors = Vectors(vector_path).vectors
            self.word_embeddings = self.word_embeddings.from_pretrained(word_vectors, freeze=not non_static).to(DEVICE)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.to(DEVICE)).to(DEVICE)

        return embeds


class BiLSTMWordEmbeddingLookup(nn.Module):

    def __init__(self, vocabulary_size, word_embedding_dim, hidden_dim, num_layers, dropout, vector_path=None, non_static=False):
        super(BiLSTMWordEmbeddingLookup, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim

        self.output_dim = hidden_dim

        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.word_embedding_dim).to(DEVICE)
        if vector_path:
            logger.info('logging word vectors from {}'.format(vector_path))
            word_vectors = Vectors(vector_path).vectors
            self.word_embeddings = self.word_embeddings.from_pretrained(word_vectors, freeze=not non_static).to(DEVICE)

        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim // 2, bidirectional=True, num_layers=num_layers, dropout=dropout).to(DEVICE)

        self.hidden = self.init_hidden()

    def forward(self, sentence):

        # word embeddings
        embeddings = self.word_embeddings(sentence)

        # lstm hidden
        # self.hidden = self.init_hidden()
        lstm_hiddens, self.hidden = self.lstm(embeddings, self.hidden)
        # print(lstm_hiddens.shape)
        return lstm_hiddens

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()
