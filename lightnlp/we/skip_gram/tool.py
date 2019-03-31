import torch
from torchtext.data import Field, Iterator
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import jieba

from ...base.tool import Tool
from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG

from .utils.dataset import SkipGramDataset

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


WORD = Field(tokenize=lambda x: [x], batch_first=True)
Fields = [
            ('context', WORD),
            ('target', WORD)
        ]


def default_tokenize(sentence):
    return list(jieba.cut(sentence))


class SkipGramTool(Tool):
    def get_dataset(self, path: str, fields=Fields):
        logger.info('loading dataset from {}'.format(path))
        cbow_dataset = SkipGramDataset(path, fields)
        logger.info('succeed loading dataset')
        return cbow_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        WORD.build_vocab(*dataset)
        logger.info('succeed building word vocab')
        return WORD.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('succeed loading vectors')
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
        assert texts.size(0) == len(labels)
        vec_predict = model(texts)
        soft_predict = torch.softmax(vec_predict, dim=1)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
        labels = labels.view(-1).cpu().data.numpy()
        return metric_func(predict_index, labels, average='micro')


skip_gram_tool = SkipGramTool()
