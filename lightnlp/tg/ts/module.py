import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import TSConfig, TSSeq2Seq
from .tool import ts_tool, light_tokenize, TEXT


class TS(Module):
    def __init__(self):
        self._model = None
        self._word_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = ts_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ts_tool.get_dataset(dev_path)
            word_vocab = ts_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab = ts_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        train_iter = ts_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = TSConfig(word_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        tsseq2seq = TSSeq2Seq(config)
        self._model = tsseq2seq
        optim = torch.optim.Adam(tsseq2seq.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            tsseq2seq.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                optim.zero_grad()
                src, src_lens = item.text
                trg, trg_lens = item.summarization
                output = tsseq2seq(src, src_lens, trg)
                output = output[1:].contiguous()
                output = output.view(-1, output.shape[-1])
                trg = trg.transpose(1, 0)
                trg = trg[1:].contiguous()
                trg = trg.view(-1)
                item_loss = F.cross_entropy(output, trg, ignore_index=TEXT.vocab.stoi['<pad>'])
                acc_loss += item_loss.item()
                item_loss.backward()
                clip_grad_norm_(tsseq2seq.parameters(), config.clip)
                optim.step()
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            print('epoch:{}, acc_loss:{}'.format(epoch, acc_loss))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        tsseq2seq.save()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = TSConfig.load(save_path)
        tsseq2seq = TSSeq2Seq(config)
        tsseq2seq .load()
        self._model = tsseq2seq
        self._word_vocab = config.word_vocab

    def test(self, test_path):
        test_dataset = ts_tool.get_dataset(test_path)
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset, batch_size=DEFAULT_CONFIG['batch_size']):
        self._model.eval()
        dev_score_list = []
        dev_iter = ts_tool.get_iterator(dev_dataset, batch_size=batch_size)
        for dev_item in tqdm(dev_iter):
            src, src_lens = dev_item.text
            trg, trg_lens = dev_item.summarization
            item_score = ts_tool.get_score(self._model, src, src_lens, trg)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def predict(self, text: str, max_len=20):
        self._model.eval()
        text_list = light_tokenize(text)
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text_list])
        vec_text = vec_text.reshape(1, -1).to(DEVICE)
        len_text = torch.tensor([len(vec_text)]).to(DEVICE)
        sos = torch.tensor([self._word_vocab.stoi['<sos>']]).to(DEVICE)
        output = self._model.predict(vec_text, len_text, sos, max_len).squeeze(1)
        soft_predict = torch.softmax(output, dim=1)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
        predict_sentence = [self._word_vocab.itos[x] for x in predict_index]
        predict_prob = predict_prob.cpu().data.tolist()
        result_sentence = []
        result_score = 1.0
        for i in range(1, len(predict_sentence)):
            if predict_sentence[i] != '<eos>':
                result_sentence.append(predict_sentence[i])
                result_score *= predict_prob[i]
            else:
                break
        return ''.join(result_sentence), result_score
