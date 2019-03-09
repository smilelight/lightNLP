import torch
from tqdm import tqdm
import torch.nn.functional as F

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import LMConfig, RNNLM
from .tool import lm_tool, light_tokenize, TEXT


class LM(Module):
    def __init__(self):
        self._model = None
        self._word_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = lm_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = lm_tool.get_dataset(dev_path)
            word_vocab = lm_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab = lm_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        train_iter = lm_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'], bptt_len=DEFAULT_CONFIG['bptt_len'])
        config = LMConfig(word_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        rnnlm = RNNLM(config)
        self._model = rnnlm
        optim = torch.optim.Adam(rnnlm.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            rnnlm.train()
            acc_loss = 0
            for fuck in tqdm(train_iter):
                optim.zero_grad()
                logits = rnnlm(fuck.text)
                item_loss = F.cross_entropy(logits, fuck.target.view(-1))
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            print('epoch:{}, acc_loss:{}'.format(epoch, acc_loss))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        rnnlm.save()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = LMConfig.load(save_path)
        rnnlm = RNNLM(config)
        rnnlm .load()
        self._model = rnnlm
        self._word_vocab = config.word_vocab

    def test(self, test_path):
        test_dataset = lm_tool.get_dataset(test_path)
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = lm_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'],
                                  bptt_len=DEFAULT_CONFIG['bptt_len'])
        for dev_item in tqdm(dev_iter):
            item_score = lm_tool.get_score(self._model, dev_item.text, dev_item.target)
            dev_score_list.append(item_score)
        # print(dev_score_list)
        return sum(dev_score_list) / len(dev_score_list)

    def _predict_next_word_max(self, sentence_list: list):
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        pred_prob, pred_index = torch.max(torch.softmax(self._model(test_item)[-1], dim=0).cpu().data, dim=0)
        pred_word = TEXT.vocab.itos[pred_index]
        pred_prob = pred_prob.item()
        return pred_word, pred_prob

    def _predict_next_word_sample(self, sentence_list: list):
        # 进行分布式采样，以获得随机结果
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        pred_index = torch.multinomial(torch.softmax(self._model(test_item)[-1], dim=0).cpu().data, 1)
        pred_word = self._word_vocab.itos[pred_index]
        return pred_word

    def _predict_next_word_topk(self, sentence_list: list, topK=5):
        # 获取topK个next个词的可能取值和对应概率
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        predict_softmax = torch.softmax(self._model(test_item)[-1], dim=0).cpu().data
        topK_prob, topK_index = torch.topk(predict_softmax, topK)
        topK_prob = topK_prob.tolist()
        topK_vocab = [self._word_vocab.itos[x] for x in topK_index]
        return list(zip(topK_vocab, topK_prob))

    def _predict_next_word_prob(self, sentence_list: list, next_word: str):
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        predict_prob = torch.softmax(self._model(test_item)[-1], dim=0).cpu().data
        next_word_index = self._word_vocab.stoi[next_word]
        return predict_prob[next_word_index]

    def next_word(self, sentence: str, next_word: str):
        self._model.eval()
        temp_str = [x for x in light_tokenize(sentence)]
        predict_prob = self._predict_next_word_prob(temp_str, next_word)
        return predict_prob.item()

    def _next_word_score(self, sentence: str, next_word: str):
        self._model.eval()
        temp_str = [x for x in light_tokenize(sentence)]
        predict_prob = self._predict_next_word_prob(temp_str, next_word)
        return torch.log10(predict_prob).item()

    def next_word_topk(self, sentence: str, topK=5):
        self._model.eval()
        return self._predict_next_word_topk(sentence, topK)

    def sentence_score(self, sentence: str):
        self._model.eval()
        total_score = 0
        assert len(sentence) > 1
        for i in range(1, len(sentence)):
            temp_score = self._next_word_score(sentence[:i], sentence[i])
            total_score += temp_score
        return total_score

    def _predict_sentence(self, sentence: str, gen_len=30):
        results = []
        temp_str = [x for x in light_tokenize(sentence)]
        for i in range(gen_len):
            temp_result = self._predict_next_word_sample(temp_str)
            results.append(temp_result)
            temp_str.append(temp_result)
        return results

    def generate_sentence(self, sentence: str, gen_len=30):
        self._model.eval()
        results = self._predict_sentence(sentence, gen_len)
        predict_sen = ''.join([x for x in results])
        return sentence + predict_sen
