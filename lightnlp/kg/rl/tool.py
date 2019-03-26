import re
import torch
from torchtext.data import TabularDataset, Field, Iterator
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from ...base.tool import Tool
from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def light_tokenize(text):
    return [text]


ENTITY = Field(tokenize=light_tokenize, batch_first=True)
RELATION = Field(tokenize=light_tokenize, batch_first=True)
Fields = [
            ('head', ENTITY),
            ('rel', RELATION),
            ('tail', ENTITY)
        ]


class RLTool(Tool):
    def get_dataset(self, path: str, fields=Fields, file_type='csv', skip_header=False):
        logger.info('loading dataset from {}'.format(path))
        rl_dataset = TabularDataset(path, format=file_type, fields=fields, skip_header=skip_header)
        logger.info('successed loading dataset')
        return rl_dataset

    def get_vocab(self, *dataset):
        logger.info('building entity vocab...')
        ENTITY.build_vocab(*dataset)
        logger.info('successed building entity vocab')
        logger.info('building relation vocab...')
        RELATION.build_vocab(*dataset)
        logger.info('successed building relation vocab')
        return ENTITY.vocab, RELATION.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE,
                     sort_key=lambda x: len(x.text)):
        return Iterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key)

    def get_score(self, model, texts, labels, score_type='f1'):
        metrics_map = {
            'f1': f1_score,
            'p': precision_score,
            'r': recall_score,
            'acc': accuracy_score
        }
        metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
        assert len(texts) == len(labels)
        vec_predict = model(texts)
        soft_predict = torch.softmax(vec_predict, dim=1)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
        # print('prob', predict_prob)
        # print('index', predict_index)
        # print('labels', labels)
        labels = labels.view(-1).cpu().data.numpy()
        return metric_func(predict_index, labels, average='micro')


rl_tool = RLTool()
