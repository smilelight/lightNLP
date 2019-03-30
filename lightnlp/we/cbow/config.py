from ...base.config import DEVICE


class Feature:
    normal = 'Normal'
    negative_sampling = 'Negative_sampling'
    hierarchical_softmax = 'Hierarchical_Softmax'


DEFAULT_CONFIG = {
    'lr': 0.005,
    'epoch': 30,
    'lr_decay': 0.05,
    'batch_size': 128,
    'embedding_dim': 300,
    'vocabulary_size': 0,
    'word_vocab': None,
    'save_path': './saves',
    'window_size': 3,
    'neg_num': 3,
    'feature': Feature.hierarchical_softmax
}
