from ...base.config import DEVICE
DEFAULT_CONFIG = {
    'lr': 0.01,
    'epoch': 10,
    'lr_decay': 0.05,
    'batch_size': 128,
    'bptt_len': 16,
    'dropout': 0.0,
    'static': False,
    'non_static': False,
    'embedding_dim': 300,
    'num_layers': 1,
    'pad_index': 1,
    'vector_path': '',
    'tag_num': 0,
    'vocabulary_size': 0,
    'word_vocab': None,
    'tag_vocab': None,
    'save_path': './saves'
}