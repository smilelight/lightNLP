import torch
from torchtext.data import Dataset, Field, BucketIterator, ReversibleField
from torchtext.vocab import Vectors
from torchtext.datasets import SequenceTaggingDataset

from ..config import DEVICE, DEFAULT_CONFIG
from .log import logger


def light_tokenize(sequence: str):
    return [sequence]


TEXT = Field(sequential=True, tokenize=light_tokenize)
TAG = ReversibleField(sequential=True, tokenize=light_tokenize, is_target=True, unk_token=None)
Fields = [('text', TEXT), ('tag', TAG)]


def get_dataset(path: str, fields=Fields, separator=' '):
    logger.info('loading dataset from {}'.format(path))
    st_dataset = SequenceTaggingDataset(path, fields=fields, separator=separator)
    logger.info('successed loading dataset')
    return st_dataset


def get_vocab(*dataset):
    logger.info('building word vocab...')
    TEXT.build_vocab(*dataset)
    logger.info('successed building word vocab')
    logger.info('building tag vocab...')
    TAG.build_vocab(*dataset)
    logger.info('successed building tag vocab')
    return TEXT.vocab, TAG.vocab


def get_vectors(path: str):
    logger.info('loading vectors from {}'.format(path))
    vectors = Vectors(path)
    logger.info('successed loading vectors')
    return vectors


def get_iterator(dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE, sort_key=lambda x: len(x.text)):
    return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key)


