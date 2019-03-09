import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DEVICE, DEFAULT_CONFIG, Actions
from ..utils import vectors


class MLPCombinerNetwork(nn.Module):

    def __init__(self, embedding_dim):
        super(MLPCombinerNetwork, self).__init__()

        self.linear1 = nn.Linear(embedding_dim*2, embedding_dim).to(DEVICE)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim).to(DEVICE)

    def forward(self, head_embed, modifier_embed):
        input_vec = vectors.concat_and_flatten((head_embed, modifier_embed))
        temp_vec = self.linear1(input_vec)
        temp_vec = torch.tanh(temp_vec)
        result = self.linear2(temp_vec)
        return result


class LSTMCombinerNetwork(nn.Module):

    def __init__(self, embedding_dim, num_layers, dropout):
        super(LSTMCombinerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_cuda = False

        self.linear = nn.Linear(self.embedding_dim*2, self.embedding_dim).to(DEVICE)
        self.hidden_dim = self.embedding_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, dropout=dropout).to(DEVICE)

        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)

        return h0, c0

    def forward(self, head_embed, modifier_embed):
        input_vec = vectors.concat_and_flatten((head_embed, modifier_embed))
        temp_vec = self.linear(input_vec).view(1, 1, -1)
        
        lstm_hiddens, self.hidden = self.lstm(temp_vec, self.hidden)
        return lstm_hiddens[-1]

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()
