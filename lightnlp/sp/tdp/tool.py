import torch
from torchtext.data import Dataset, Field, BucketIterator, ReversibleField
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from ...base.tool import Tool
from ...utils.log import logger
from .utils.dataset import TransitionDataset
from .config import DEVICE, DEFAULT_CONFIG, END_OF_INPUT_TOK, ROOT_TOK, NULL_STACK_TOK

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def light_tokenize(sequence: str):
    return sequence.split()


def action_tokenize(sequence: str):
    return [sequence]


TEXT = Field(sequential=True, tokenize=light_tokenize, eos_token=END_OF_INPUT_TOK, pad_token=None)
ACTION = ReversibleField(
    sequential=True, tokenize=action_tokenize, is_target=True, unk_token=None, pad_token=None)

Fields = [('text', TEXT), ('action', ACTION)]


class TDPTool(Tool):
    def get_dataset(self, path: str, fields=Fields, separator=' ||| '):
        logger.info('loading dataset from {}'.format(path))
        tdp_dataset = TransitionDataset(path, fields=fields, separator=separator)
        logger.info('successed loading dataset')
        return tdp_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        TEXT.build_vocab(*dataset, specials=[ROOT_TOK, NULL_STACK_TOK])
        logger.info('successed building word vocab')
        logger.info('building tag vocab...')
        ACTION.build_vocab(*dataset)
        logger.info('successed building tag vocab')
        return TEXT.vocab, ACTION.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE):
        return BucketIterator(dataset, batch_size=batch_size, device=device)

    def get_score(self, model, x, y, score_type='f1'):
        metrics_map = {
            'f1': f1_score,
            'p': precision_score,
            'r': recall_score,
            'acc': accuracy_score
        }
        metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
        outputs, dep_graph, actions_done = model(x)
        assert len(actions_done) == len(y)
        predict_y = actions_done
        true_y = y.cpu().view(-1).tolist()
        # print(actions_done, y)
        # print(actions_done)
        # print(true_y)
        return metric_func(predict_y, true_y, average='micro')


tdp_tool = TDPTool()
