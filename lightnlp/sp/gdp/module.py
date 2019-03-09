import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import Config, BiaffineParser
from .tool import gdp_tool, WORD, POS, REF, ROOT
from .utils.metric import AttachmentMethod

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class GDP(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._pos_vocab = None
        self._ref_vocab = None
        self._pad_index = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = gdp_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = gdp_tool.get_dataset(dev_path)
            word_vocab, pos_vocab, head_vocab, ref_vocab = gdp_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, pos_vocab, head_vocab, ref_vocab = gdp_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._pos_vocab = pos_vocab
        self._ref_vocab = ref_vocab
        train_iter = gdp_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, pos_vocab, ref_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        biaffine_parser = BiaffineParser(config)
        self._model = biaffine_parser
        self._pad_index = config.pad_index
        optim = torch.optim.Adam(biaffine_parser.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2), eps=config.epsilon)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                                      lr_lambda=lambda x: config.decay ** (x / config.decay_steps))
        for epoch in range(config.epoch):
            biaffine_parser.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                biaffine_parser.zero_grad()
                words = item.word
                tags = item.pos
                refs = item.ref
                arcs = item.head
                # print('arcs', arcs)
                mask = words.ne(config.pad_index)
                mask[:, 0] = 0
                s_arc, s_rel = self._model(words, tags)
                s_arc, s_rel = s_arc[mask], s_rel[mask]
                # print('s_arc', s_arc)
                # print(s_arc.shape)
                gold_arcs, gold_rels = arcs[mask], refs[mask]
                # print(gold_arcs)
                # print(gold_arcs.shape)
                item_loss = self._get_loss(s_arc, s_rel, gold_arcs, gold_rels)
                acc_loss += item_loss.cpu().item()
                item_loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optim.step()
                scheduler.step()
            acc_loss /= len(train_iter)
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score, dev_metric = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                logger.info('metric:{}'.format(dev_metric))

            # adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        biaffine_parser.save()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        biaffine_parser = BiaffineParser(config)
        biaffine_parser.load()
        self._model = biaffine_parser
        self._word_vocab = config.word_vocab
        self._pos_vocab = config.pos_vocab
        self._ref_vocab = config.ref_vocab
        self._pad_index = config.pad_index
        self._check_vocab()

    def test(self, test_path):
        test_dataset = gdp_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _check_vocab(self):
        if not hasattr(WORD, 'vocab'):
            WORD.vocab = self._word_vocab
        if not hasattr(POS, 'vocab'):
            POS.vocab = self._pos_vocab
        if not hasattr(REF, 'vocab'):
            REF.vocab = self._ref_vocab

    def _validate(self, dev_dataset):
        self._model.eval()
        loss, metric = 0, AttachmentMethod()
        dev_iter = gdp_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            mask = dev_item.word.ne(self._pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self._model(dev_item.word, dev_item.pos)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = dev_item.head[mask], dev_item.ref[mask]
            pred_arcs, pred_rels = self._decode(s_arc, s_rel)
            loss += self._get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(dev_iter)
        return loss, metric

    def predict(self, word_list: list, pos_list: list):
        self._model.eval()
        assert len(word_list) == len(pos_list)
        word_list.insert(0, ROOT)
        pos_list.insert(0, ROOT)
        vec_word = WORD.numericalize([word_list]).to(DEVICE)
        vec_pos = POS.numericalize([pos_list]).to(DEVICE)
        mask = vec_word.ne(self._pad_index)
        s_arc, s_rel = self._model(vec_word, vec_pos)
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        pred_arcs, pred_rels = self._decode(s_arc, s_rel)
        pred_arcs = pred_arcs.cpu().tolist()
        pred_arcs[0] = 0
        pred_rels = [self._ref_vocab.itos[rel] for rel in pred_rels]
        pred_rels[0] = ROOT
        return pred_arcs, pred_rels

    def _get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        arc_loss = F.cross_entropy(s_arc, gold_arcs)
        rel_loss = F.cross_entropy(s_rel, gold_rels)
        loss = arc_loss + rel_loss
        return loss

    def _decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels
