from ...base.config import DEVICE
from ...base.model import BaseConfig
from .models import *
DEFAULT_CONFIG = {
    'lr': 0.02,
    'epoch': 1000,
    'lr_decay': 0.05,
    'batch_size': 128,
    'dropout': 0.5,
    'static': False,
    'non_static': False,
    'embedding_dim': 300,
    'num_layers': 2,
    'pad_index': 1,
    'score_func': 'l2',
    'rel_num': 0,
    'entity_num': 0,
    'entity_vocab': None,
    'rel_vocab': None,
    'loss_margin': 2.0,
    'save_path': './saves',
    'model_type': 'TransE'
}

MODELS = {
    'TransE': TransE
}


class Config(BaseConfig):
    def __init__(self, entity_vocab, rel_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.entity_vocab = entity_vocab
        self.rel_vocab = rel_vocab
        self.entity_num = len(self.entity_vocab)
        self.rel_num = len(self.rel_vocab)
        for name, value in kwargs.items():
            setattr(self, name, value)
