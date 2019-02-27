import torch
from torchtext.data import ReversibleField, BPTTIterator, Dataset
from torchtext.datasets import LanguageModelingDataset
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from ...base.tool import Tool
from ...utils.log import logger
from .config import DEVICE, DEFAULT_CONFIG

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def light_tokenize(sequence: str):
    return [x for x in sequence.strip()]


TEXT = ReversibleField(sequential=True, tokenize=light_tokenize)


class LMTool(Tool):
    def tokenize(self, sequence: str):
        return [x for x in sequence.strip()]

    def get_dataset(self, path: str, field=TEXT, newline_eos=False):
        logger.info('loading dataset from {}'.format(path))
        lm_dataset = LanguageModelingDataset(path, text_field=field, newline_eos=newline_eos)
        logger.info('successed loading dataset')
        return lm_dataset

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

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'],
                     bptt_len=DEFAULT_CONFIG['bptt_len'], device=DEVICE):
        return BPTTIterator(dataset, batch_size=batch_size,
                            bptt_len=bptt_len, device=device)

    def get_score(self, model, x, y, score_type='f1'):
        metrics_map = {
            'f1': f1_score,
            'p': precision_score,
            'r': recall_score,
            'acc': accuracy_score
        }
        metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
        vec_x = x
        predict_y = model(vec_x.view(-1, 1).to(DEVICE))
        predict_index = torch.max(torch.softmax(predict_y, dim=1).cpu().data, dim=1)[1]
        predict_index = predict_index.data.numpy()
        true_y = y.view(-1).cpu().data.numpy()
        assert len(true_y) == len(predict_index)
        return metric_func(predict_index, true_y, average='micro')


lm_tool = LMTool()
