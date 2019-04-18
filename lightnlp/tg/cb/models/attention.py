import torch.nn as nn
import torch.nn.functional as F
import torch

from ..config import DEVICE, DEFAULT_CONFIG


class Attention(nn.Module):
    """
    several score types like dot,general and concat
    """
    def __init__(self, method='dot', hidden_size=None):
        super(Attention, self).__init__()
        self.method = method
        if self.method != 'dot':
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.W = nn.Linear(hidden_size, hidden_size)
            elif self.method == 'concat':
                self.W = nn.Linear(self.hidden_size * 2, hidden_size)
                self.v = nn.Parameter(torch.rand(1, hidden_size))  # 此处定义为Linear也可以
                nn.init.xavier_normal_(self.v.data)

    def forward(self, query, key, value, mask=None, dropout=0):
        if self.method == 'general':
            scores = self.general(query, key)
        elif self.method == 'concat':
            scores = self.concat(query, key)
        else:
            scores = self.dot(query, key)

        # normalize
        # scores = scores / math.sqrt(query.size(-1))

        # mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax
        p_attn = F.softmax(scores, dim=-1)

        # dropout
        if not dropout:
            p_attn = F.dropout(p_attn, dropout)

        return torch.matmul(p_attn, value), p_attn

    def dot(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1))
        return scores

    def general(self, query, key):
        scores = torch.matmul(self.W(query), key.transpose(-2, -1))
        return scores

    def concat(self, query, key):
        scores = torch.cat((query.expand(-1, key.size(1), -1), key), dim=2)
        scores = self.W(scores)
        scores = F.tanh(scores)
        scores = torch.matmul(scores, self.v.t()).transpose(-2, -1)
        return scores


if __name__ == '__main__':
    dim = 10
    key_size = 5
    batch_size = 4
    q = torch.rand((batch_size, 1, dim))
    k = torch.rand(batch_size, key_size, dim)
    v = k
    # dot
    # attention = Attention()
    # general
    # attention = Attention('general', dim)
    # concat
    attention = Attention('concat', dim)
    output, score = attention(q, k, v)
    print(output)
    print(output.shape)
    print(score)
    print(score.shape)
