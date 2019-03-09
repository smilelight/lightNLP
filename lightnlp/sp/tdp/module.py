import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import Config, TransitionParser
from .tool import tdp_tool, TEXT, ACTION, light_tokenize
from .utils.parser_state import DepGraphEdge

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class TDP(Module):
    """
    """
    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._action_vocab = None
    
    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = tdp_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = tdp_tool.get_dataset(dev_path)
            word_vocab, action_vocab = tdp_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, action_vocab = tdp_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._action_vocab = action_vocab
        train_iter = tdp_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, action_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        trainsition_parser = TransitionParser(config)
        self._model = trainsition_parser
        optim = torch.optim.Adam(trainsition_parser.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            trainsition_parser.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self._model.refresh()
                optim.zero_grad()
                outputs, dep_graph, actions_done = self._model(item.text)
                item_loss = 0
                for step_output, step_action in zip(outputs, item.action):
                    # print(step_output)
                    # print(step_action)
                    item_loss += F.cross_entropy(step_output, step_action)
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            acc_loss /= len(train_iter)
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        trainsition_parser.save()

    def predict(self, text):
        self._model.eval()
        sentences = light_tokenize(text)
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in sentences])
        outputs, dep_graph, actions_done = self._model(vec_text.view(-1, 1).to(DEVICE))
        # tag_predict = [self._action_vocab.itos[i] for i in vec_predict]
        # return iob_ranges([x for x in text], tag_predict)
        results = set()
        for edge in dep_graph:
            results.add(DepGraphEdge((self._word_vocab.itos[edge.head[0]], edge.head[1]),
                                     (self._word_vocab.itos[edge.modifier[0]], edge.modifier[1])))
        return results

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        trainsition_parser = TransitionParser(config)
        trainsition_parser.load()
        self._model = trainsition_parser
        self._word_vocab = config.word_vocab
        self._action_vocab = config.action_vocab
        self._check_vocab()
    
    def test(self, test_path):
        test_dataset = tdp_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))
    
    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = tdp_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            self._model.refresh()
            item_score = tdp_tool.get_score(self._model, dev_item.text, dev_item.action)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def _check_vocab(self):
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        if not hasattr(ACTION, 'vocab'):
            ACTION.vocab = self._action_vocab
