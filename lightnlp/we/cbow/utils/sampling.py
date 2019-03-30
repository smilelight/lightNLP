import torch
from torchtext.vocab import Vocab


class Sampling(object):
    def __init__(self, vocab: Vocab, weight=0.75):
        self.vocab = vocab
        self.weight = weight
        self.weighted_list = [self.vocab.freqs[s]**self.weight for s in self.vocab.itos]

    def sampling(self, num):
        return torch.multinomial(torch.tensor(self.weighted_list), num).tolist()
