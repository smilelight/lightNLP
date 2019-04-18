import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from ..config import DEVICE, DEFAULT_CONFIG


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2, method='dot'):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size).to(DEVICE)
        self.dropout = nn.Dropout(dropout, inplace=True).to(DEVICE)
        self.attention = Attention(method, hidden_size).to(DEVICE)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout, batch_first=True).to(DEVICE)
        self.out = nn.Linear(hidden_size * 2, output_size).to(DEVICE)

    def forward(self, word, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(word).unsqueeze(1)  # (B,1,N)
        embedded = self.dropout(embedded)

        # Calculate attention weights and apply to encoder outputs
        context, attn_weights = self.attention(last_hidden[-1].unsqueeze(1), encoder_outputs, encoder_outputs)
        # Combine embedded input word and attended context, run through RNN
        context = F.relu(context)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)  # (B,1,N) -> (B,N)
        context = context.squeeze(1)
        output = torch.cat((output, context), 1)
        output = self.out(output)
        # output = F.softmax(output, dim=1)
        return output, hidden, attn_weights
