import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG, Config, MODELS
from .tool import rl_tool, light_tokenize, ENTITY, RELATION
from .utils.get_neg_batch import get_neg_batch

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class RL(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._entity_vocab = None
        self._relation_vocab = None

    def train(self, train_path, model_type=DEFAULT_CONFIG['model_type'], save_path=DEFAULT_CONFIG['save_path'], dev_path=None, **kwargs):
        train_dataset = rl_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = rl_tool.get_dataset(dev_path)
            entity_vocab, relation_vocab = rl_tool.get_vocab(train_dataset, dev_dataset)
        else:
            entity_vocab, relation_vocab = rl_tool.get_vocab(train_dataset)
        self._entity_vocab = entity_vocab
        self._relation_vocab = relation_vocab
        train_iter = rl_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(entity_vocab, relation_vocab, save_path=save_path, **kwargs)
        if model_type in MODELS:
            model = MODELS[model_type](config)
        else:
            raise Exception('there is no model named {}! please check the name carefully'.format(model_type))
        # print(self._model)
        self._model = model
        optim = torch.optim.Adam(self._model.parameters(), lr=config.lr)
        # criterion = nn.MarginRankingLoss(config.loss_margin, reduction='sum')
        criterion = nn.MarginRankingLoss(config.loss_margin, reduction='mean')
        for epoch in range(config.epoch):
            self._model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                optim.zero_grad()
                # item.head = item.head.reshape(item.head.size(0), -1)
                # item.rel = item.rel.reshape(item.rel.size(0), -1)
                # item.tail = item.tail.reshape(item.tail.size(0), -1)
                neg_head, neg_tail = get_neg_batch(item.head, item.tail, self._model.entity_num)
                pos_score = self._model(item.head, item.rel, item.tail)
                neg_score = self._model(neg_head, item.rel, neg_tail)
                item_loss = criterion(neg_score, pos_score, torch.ones_like(pos_score))
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        self._model.save()

    def predict(self, head: str, rel: str, tail:str):
        self._model.eval()
        if head not in self._entity_vocab.stoi:
            print('there is not entity named:{} in database!'.format(head))
            return None
        if rel not in self._relation_vocab.stoi:
            print('there is not relation named:{} in database!'.format(rel))
            return None
        if tail not in self._entity_vocab.stoi:
            print('there is not entity named:{} in database!'.format(tail))
            return None
        head_id = torch.tensor([self._entity_vocab.stoi[head]]).view((-1, 1)).to(DEVICE)
        rel_id = torch.tensor([self._relation_vocab.stoi[rel]]).view((-1, 1)).to(DEVICE)
        tail_id = torch.tensor([self._entity_vocab.stoi[tail]]).view((-1, 1)).to(DEVICE)
        score = torch.exp(- self._model(head_id, rel_id, tail_id))[0].item()
        return score

    def predict_head(self, rel: str, tail: str, topk=3):
        self._model.eval()
        if rel not in self._relation_vocab.stoi:
            print('there is not relation named:{} in database!'.format(rel))
            return None
        if tail not in self._entity_vocab.stoi:
            print('there is not entity named:{} in database!'.format(tail))
            return None
        rel_id = torch.tensor([self._relation_vocab.stoi[rel]] * self._model.entity_num).view((-1, 1)).to(DEVICE)
        tail_id = torch.tensor([self._entity_vocab.stoi[tail]] * self._model.entity_num).view((-1, 1)).to(DEVICE)
        candidate_heads = torch.arange(self._model.entity_num).view((-1, 1)).to(DEVICE)
        head_scores = torch.exp(- self._model(candidate_heads, rel_id, tail_id))
        topk_scores, topk_index = torch.topk(head_scores, topk)
        topk_scores = topk_scores.tolist()
        topk_entities = [self._entity_vocab.itos[i] for i in topk_index]
        return list(zip(topk_entities, topk_scores))

    def predict_rel(self, head, tail, topk=3):
        self._model.eval()
        if head not in self._entity_vocab.stoi:
            print('there is not entity named:{} in database!'.format(head))
            return None
        if tail not in self._entity_vocab.stoi:
            print('there is not entity named:{} in database!'.format(tail))
            return None
        head_id = torch.tensor([self._entity_vocab.stoi[head]] * self._model.rel_num).view((-1, 1)).to(DEVICE)
        tail_id = torch.tensor([self._entity_vocab.stoi[tail]] * self._model.rel_num).view((-1, 1)).to(DEVICE)
        candidate_rels = torch.arange(self._model.rel_num).view((-1, 1)).to(DEVICE)
        rel_scores = torch.exp(- self._model(head_id, candidate_rels, tail_id))
        topk_scores, topk_index = torch.topk(rel_scores, topk)
        topk_scores = topk_scores.tolist()
        topk_rels = [self._relation_vocab.itos[i] for i in topk_index.tolist()]
        return list(zip(topk_rels, topk_scores))

    def predict_tail(self, head, rel, topk=3):
        self._model.eval()
        if head not in self._entity_vocab.stoi:
            print('there is not entity named:{} in database!'.format(head))
            return None
        if rel not in self._relation_vocab.stoi:
            print('there is not relation named:{} in database!'.format(rel))
            return None
        head_id = torch.tensor([self._entity_vocab.stoi[head]] * self._model.entity_num).view((-1, 1)).to(DEVICE)
        rel_id = torch.tensor([self._relation_vocab.stoi[rel]] * self._model.entity_num).view((-1, 1)).to(DEVICE)
        candidate_tails = torch.arange(self._model.entity_num).view((-1, 1)).to(DEVICE)
        head_scores = torch.exp(- self._model(head_id, rel_id, candidate_tails))
        topk_scores, topk_index = torch.topk(head_scores, topk)
        topk_scores = topk_scores.tolist()
        topk_entities = [self._entity_vocab.itos[i] for i in topk_index]
        return list(zip(topk_entities, topk_scores))

    def load(self, save_path=DEFAULT_CONFIG['save_path'], model_type=DEFAULT_CONFIG['model_type']):
        config = Config.load(save_path)
        if model_type in MODELS:
            self._model = MODELS[model_type](config)
        else:
            raise Exception('there is no model named {}! please check the name carefully'.format(model_type))
        self._model.load()
        self._model = self._model
        self._entity_vocab = config.entity_vocab
        self._relation_vocab = config.rel_vocab
        self._check_vocab()

    def test(self, test_path):
        test_dataset = rl_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset, batch_size=DEFAULT_CONFIG['batch_size']):
        self._model.eval()
        dev_score_list = []
        dev_iter = rl_tool.get_iterator(dev_dataset, batch_size=batch_size)
        for dev_item in tqdm(dev_iter):
            item_score = torch.exp(- self._model(dev_item.head, dev_item.rel, dev_item.tail))
            # item_score = self._model(dev_item.head, dev_item.rel, dev_item.tail)
            dev_score_list.extend(item_score.cpu().tolist())
        return sum(dev_score_list) / len(dev_score_list)

    def _check_vocab(self):
        if not hasattr(ENTITY, 'vocab'):
            ENTITY.vocab = self._entity_vocab
        if not hasattr(RELATION, 'vocab'):
            RELATION.vocab = self._relation_vocab
