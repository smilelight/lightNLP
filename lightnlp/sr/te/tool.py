import re
import torch
from torchtext.data import TabularDataset, Field, Iterator, Dataset, BucketIterator
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from ...base.tool import Tool
from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9，。？：！；“”]')


def light_tokenize(text):
    text = regex.sub(' ', text)
    return [word for word in text if word.strip()]


TEXT = Field(lower=True, tokenize=light_tokenize, fix_length=DEFAULT_CONFIG['fix_length'])
LABEL = Field(sequential=False, unk_token=None)
Fields = [
            ('texta', TEXT),
            ('textb', TEXT),
            ('label', LABEL)
        ]


class TETool(Tool):
    def get_dataset(self, path: str, fields=Fields, file_type='tsv', skip_header=True):
        logger.info('loading dataset from {}'.format(path))
        te_dataset = TabularDataset(path, format=file_type, fields=fields, skip_header=skip_header)
        logger.info('successed loading dataset')
        return te_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        TEXT.build_vocab(*dataset)
        logger.info('successed building word vocab')
        logger.info('building label vocab...')
        LABEL.build_vocab(*dataset)
        logger.info('successed building label vocab')
        return TEXT.vocab, LABEL.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE,
                 sort_key=lambda x: len(x.texta)):
        return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key)

    def get_score(self, model, texta, textb, labels, score_type='f1'):
        metrics_map = {
            'f1': f1_score,
            'p': precision_score,
            'r': recall_score,
            'acc': accuracy_score
        }
        metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
        assert texta.size(1) == textb.size(1) == len(labels)
        vec_predict = model(texta, textb)
        soft_predict = torch.softmax(vec_predict, dim=1)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
        # print('prob', predict_prob)
        # print('index', predict_index)
        # print('labels', labels)
        labels = labels.view(-1).cpu().data.numpy()
        return metric_func(predict_index, labels, average='micro')


te_tool = TETool()