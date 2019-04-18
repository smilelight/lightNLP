import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..config import DEVICE, DEFAULT_CONFIG


class Encoder(nn.Module):
    """
    basic GRU encoder
    """
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size).to(DEVICE)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True,
                          dropout=dropout, bidirectional=True).to(DEVICE)

    def forward(self, sentences, lengths, hidden=None):
        embedded = self.embed(sentences)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden
