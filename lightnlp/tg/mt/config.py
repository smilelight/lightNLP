from ...base.config import DEVICE
DEFAULT_CONFIG = {
    'lr': 0.02,
    'epoch': 300,
    'lr_decay': 0.05,
    'batch_size': 4,
    'dropout': 0.5,
    'static': False,
    'non_static': False,
    'source_embedding_dim': 100,
    'target_embedding_dim': 100,
    'hidden_dim': 100,
    'clip': 10,
    'num_layers': 2,
    'source_vector_path': '',
    'target_vector_path': '',
    'source_vocabulary_size': 0,
    'target_vocabulary_size': 0,
    'source_word_vocab': None,
    'target_word_vocab': None,
    'save_path': './saves',
    'method': 'dot',
    'teacher_forcing_ratio': 0.5
}
