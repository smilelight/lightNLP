import re

import jieba
import torch
from torchtext.data import TabularDataset, Field, Iterator
from torchtext.vocab import Vectors

from ..config import DEVICE, DEFAULT_CONFIG
from .log import logger

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9，。？：！；“”]')

def _word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


# TEXT = Field(lower=True, tokenize=_word_cut, batch_first=True)
# LABEL = Field(sequential=False, unk_token=None, batch_first=True)
TEXT = Field(lower=True, tokenize=_word_cut)
LABEL = Field(sequential=False, unk_token=None)
Fields = [
            ('index', None),
            ('label', LABEL),
            ('text', TEXT)
        ]

def get_dataset(path: str, fields=Fields, file_type='tsv', skip_header=True):
    logger.info('loading dataset from {}'.format(path))
    st_dataset = TabularDataset(path, format= file_type, fields=fields, skip_header=skip_header)
    logger.info('successed loading dataset')
    return st_dataset


def get_vocab(*dataset):
    logger.info('building word vocab...')
    TEXT.build_vocab(*dataset)
    logger.info('successed building word vocab')
    logger.info('building label vocab...')
    LABEL.build_vocab(*dataset)
    logger.info('successed building label vocab')
    return TEXT.vocab, LABEL.vocab


def get_vectors(path: str):
    logger.info('loading vectors from {}'.format(path))
    vectors = Vectors(path)
    logger.info('successed loading vectors')
    return vectors


def get_iterator(dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE, sort_key=lambda x: len(x.text)):
    return Iterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key)