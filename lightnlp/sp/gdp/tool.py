import torch
from torchtext.data import Dataset, Field, BucketIterator
from torchtext.vocab import Vectors
from torchtext.datasets import SequenceTaggingDataset

from ...base.tool import Tool
from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG, ROOT

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# id, from, lemma, cpostag, postag, feats, head, depref


def light_tokenize(sequence: str):
    return [sequence]


def head_tokenize(sequence: str):
    return int(sequence)


def post_process(arr, _):
    return [[int(item) for item in arr_item] for arr_item in arr]


# WORD = Field(sequential=True, tokenize=light_tokenize, include_lengths=True)
WORD = Field(sequential=True, tokenize=light_tokenize, batch_first=True, init_token=ROOT)
POS = Field(sequential=True, tokenize=light_tokenize, unk_token=None, batch_first=True, init_token=ROOT)
HEAD = Field(sequential=True, use_vocab=False, unk_token=None, pad_token=0, postprocessing=post_process,
             batch_first=True, init_token=0)
REF = Field(sequential=True, tokenize=light_tokenize, unk_token=None, batch_first=True, init_token=ROOT)
Fields = [('id', None), ('word', WORD), ('lemma', None), ('cpostag', None), ('pos', POS),
          ('feats', None), ('head', HEAD), ('ref', REF)]


class GDPTool(Tool):
    def get_dataset(self, path: str, fields=Fields, separator='\t'):
        logger.info('loading dataset from {}'.format(path))
        gdp_dataset = SequenceTaggingDataset(path, fields=fields, separator=separator)
        logger.info('successed loading dataset')
        return gdp_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        WORD.build_vocab(*dataset)
        logger.info('successed building word vocab')
        logger.info('building pos vocab...')
        POS.build_vocab(*dataset)
        logger.info('successed building pos vocab')
        logger.info('building head vocab...')
        HEAD.build_vocab(*dataset)
        logger.info('successed building head vocab')
        logger.info('building ref vocab...')
        REF.build_vocab(*dataset)
        logger.info('successed building ref vocab')
        return WORD.vocab, POS.vocab, HEAD.vocab, REF.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE,
                     sort_key=lambda x: len(x.word), sort_within_batch=True):
        return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key,
                              sort_within_batch=sort_within_batch)

    def get_score(self):
        pass


gdp_tool = GDPTool()

