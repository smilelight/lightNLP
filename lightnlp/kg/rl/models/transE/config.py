from .....base.config import DEVICE
DEFAULT_CONFIG = {
    'lr': 0.02,
    'epoch': 30,
    'lr_decay': 0.05,
    'batch_size': 128,
    'dropout': 0.5,
    'static': False,
    'non_static': False,
    'embedding_dim': 300,
    'num_layers': 2,
    'pad_index': 1,
    'score_func': 'l1',
    'rel_num': 0,
    'entity_num': 0,
    'entity_vocab': None,
    'rel_vocab': None,
    'save_path': './saves'
}