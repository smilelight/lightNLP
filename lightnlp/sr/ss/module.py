import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import TabularDataset, Field, Iterator, Dataset, BucketIterator
from torchtext.vocab import Vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .tool import ss_tool, TEXT, LABEL
from .config import DEVICE, DEFAULT_CONFIG
from .model import Config, MaLSTM
from .utils.pad import pad_sequnce

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class SS(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._label_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = ss_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ss_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = ss_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = ss_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._label_vocab = tag_vocab
        train_iter = ss_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        malstm = MaLSTM(config)
        self._model = malstm
        optim = torch.optim.Adam(self._model.parameters(), lr=config.lr)
        loss_func = torch.nn.MSELoss().to(DEVICE)
        for epoch in range(config.epoch):
            self._model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self._model.zero_grad()
                left_text = item.texta
                right_text = item.textb
                predict_dis = self._model(left_text, right_text)
                item_loss = loss_func(predict_dis, item.label.type(torch.float32))
                acc_loss += item_loss.view(-1).cpu().item()
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))

            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        self._model.save()

    def predict(self, texta: str, textb: str):
        self._model.eval()
        pad_texta = pad_sequnce([x for x in texta], DEFAULT_CONFIG['fix_length'])
        vec_texta = torch.tensor([self._word_vocab.stoi[x] for x in pad_texta])
        pad_textb = pad_sequnce([x for x in textb], DEFAULT_CONFIG['fix_length'])
        vec_textb = torch.tensor([self._word_vocab.stoi[x] for x in pad_textb])
        vec_predict = self._model(vec_texta.view(-1, 1).to(DEVICE),
                                  vec_textb.view(-1, 1).to(DEVICE))[0]
        return vec_predict.cpu().item()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        malstm = MaLSTM(config)
        malstm.load()
        self._model = malstm
        self._word_vocab = config.word_vocab
        self._label_vocab = config.label_vocab

    def test(self, test_path):
        test_dataset = ss_tool.get_dataset(test_path)
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        if not hasattr(LABEL, 'vocab'):
            LABEL.vocab = self._label_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = ss_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            item_score = ss_tool.get_score(self._model, dev_item.texta, dev_item.textb, dev_item.label)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)
