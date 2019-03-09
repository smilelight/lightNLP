from ...base.config import DEVICE

DEFAULT_CONFIG = {
    'lr': 0.01,
    'epoch': 5,
    'lr_decay': 0.05,
    'batch_size': 1,
    'dropout': 0.5,
    'static': True,
    'non_static': False,
    'embedding_dim': 300,
    'vector_path': '',
    'class_num': 0,
    'vocabulary_size': 0,
    'word_vocab': None,
    'action_vocab': None,
    'save_path': './saves',
    'embedding_type': 'lstm',
    'embedding_lstm_layers': 1,
    'embedding_lstm_dropout': 0.0,
    'combiner': 'lstm',
    'combiner_lstm_layers': 1,
    'combiner_lstm_dropout': 0.0,
    'action_chooser': 'default',
    'feature_extractor': 'default',
    'num_features': 3,
    'word_embedding_dim': 100,
    'stack_embedding_dim': 100
}


class Actions:
    """Simple Enum for each possible parser action"""
    SHIFT = 0
    REDUCE_L = 1
    REDUCE_R = 2

    NUM_ACTIONS = 3

    action_to_ix = { "SHIFT": SHIFT,
                     "REDUCE_L": REDUCE_L,
                     "REDUCE_R": REDUCE_R }


END_OF_INPUT_TOK = "<END-OF-INPUT>"
NULL_STACK_TOK = "<NULL-STACK>"
ROOT_TOK = "<ROOT>"
