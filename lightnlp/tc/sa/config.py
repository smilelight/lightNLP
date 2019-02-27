from ...base.config import DEVICE
DEFAULT_CONFIG = {
    'lr': 0.02,
    'epoch': 5,
    'lr_decay': 0.05,
    'batch_size': 128,
    'dropout': 0.5,
    'static': False,
    'non_static': False,
    'embedding_dim': 300,
    'vector_path': '',
    'class_num': 0,
    'vocabulary_size': 0,
    'word_vocab': None,
    'tag_vocab': None,
    'save_path': './saves',
    'filter_num': 100,
    'filter_sizes': (3, 4, 5),
    'multichannel': False
}