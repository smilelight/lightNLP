from ...base.config import DEVICE

DEFAULT_CONFIG = {
    'lr': 2e-3,
    'beta_1': 0.9,
    'beta_2': 0.9,
    'epsilon': 1e-12,
    'decay': .75,
    'decay_steps': 5000,
    'epoch': 50,
    'patience': 100,
    'pad_index': 1,
    'lr_decay': 0.05,
    'batch_size': 2,
    'dropout': 0.5,
    'static': True,
    'non_static': False,
    'word_dim': 300,
    'embed_dropout': 0.33,
    'vector_path': '',
    'class_num': 0,
    'vocabulary_size': 0,
    'word_vocab': None,
    'pos_vocab': None,
    'ref_vocab': None,
    'save_path': './saves',
    'pos_dim': 100,
    'lstm_hidden': 400,
    'lstm_layers': 3,
    'lstm_dropout': 0.33,
    'mlp_arc': 500,
    'mlp_rel': 100,
    'mlp_dropout': 0.33
}

ROOT = '<ROOT>'


class Actions:
    """Simple Enum for each possible parser action"""
    SHIFT = 0
    REDUCE_L = 1
    REDUCE_R = 2

    NUM_ACTIONS = 3

    action_to_ix = { "SHIFT": SHIFT,
                     "REDUCE_L": REDUCE_L,
                     "REDUCE_R": REDUCE_R }


