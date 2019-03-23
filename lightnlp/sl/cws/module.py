import torch
from tqdm import tqdm

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import Config, BiLstmCrf
from .tool import cws_tool
from .utils.convert import bis_cws

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class CWS(Module):
    """
    """
    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None
    
    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = cws_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = cws_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = cws_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = cws_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        train_iter = cws_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = (-bilstmcrf.loss(item_text_sentences, item_text_lengths, item.tag)) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))

            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        bilstmcrf.save()

    def predict(self, text):
        self._model.eval()
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text])
        len_text = torch.tensor([len(vec_text)]).to(DEVICE)
        vec_predict = self._model(vec_text.view(-1, 1).to(DEVICE), len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        return bis_cws([x for x in text], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
    
    def test(self, test_path):
        test_dataset = cws_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))
    
    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        for dev_item in tqdm(dev_dataset):
            item_score = cws_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab, self._tag_vocab)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)
