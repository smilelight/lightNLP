import re
import torch
from torchtext.data import TabularDataset, Field, Dataset, BucketIterator
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import jieba

from ...base.tool import Tool
from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9，。？：！；“”]')


def light_tokenize(text):
    # text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


TEXT = Field(lower=True, tokenize=light_tokenize, include_lengths=True, batch_first=True, init_token='<sos>',
             eos_token='<eos>')
Fields = [
            ('query', TEXT),
            ('answer', TEXT)
        ]


class CBTool(Tool):
    def get_dataset(self, path: str, fields=Fields, file_type='tsv', skip_header=False):
        logger.info('loading dataset from {}'.format(path))
        cb_dataset = TabularDataset(path, format=file_type, fields=fields, skip_header=skip_header)
        logger.info('successed loading dataset')
        return cb_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        TEXT.build_vocab(*dataset)
        logger.info('successed building word vocab')
        return TEXT.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE,
                     sort_key=lambda x: len(x.query), sort_within_batch=True):
        return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key,
                              sort_within_batch=sort_within_batch)

    def get_score(self, model, src, src_lens, trg, score_type='f1'):
        metrics_map = {
            'f1': f1_score,
            'p': precision_score,
            'r': recall_score,
            'acc': accuracy_score
        }
        metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
        output = model(src, src_lens, trg)
        output = output[1:].contiguous()
        output = output.view(-1, output.shape[-1])
        trg = trg.transpose(1, 0)
        trg = trg[1:].contiguous()
        trg = trg.view(-1)
        soft_predict = torch.softmax(output, dim=1)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
        labels = trg.cpu().data.numpy()
        return metric_func(predict_index, labels, average='micro')


cb_tool = CBTool()
