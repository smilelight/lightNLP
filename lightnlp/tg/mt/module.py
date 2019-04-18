import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ...utils.learning import adjust_learning_rate
from ...utils.log import logger
from ...base.module import Module

from .config import DEVICE, DEFAULT_CONFIG
from .model import MTConfig, MTSeq2Seq
from .tool import mt_tool, eng_tokenize, SOURCE, TARGET


class MT(Module):
    def __init__(self):
        self._model = None
        self._source_vocab = None
        self._target_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, source_vectors_path=None,
              target_vectors_path=None, **kwargs):
        train_dataset = mt_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = mt_tool.get_dataset(dev_path)
            source_vocab, target_vocab = mt_tool.get_vocab(train_dataset, dev_dataset)
        else:
            source_vocab, target_vocab = mt_tool.get_vocab(train_dataset)
        self._source_vocab = source_vocab
        self._target_vocab = target_vocab
        train_iter = mt_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = MTConfig(source_vocab, target_vocab, source_vectors_path, target_vectors_path, save_path=save_path,
                          **kwargs)
        mtseq2seq = MTSeq2Seq(config)
        self._model = mtseq2seq
        optim = torch.optim.Adam(mtseq2seq.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            mtseq2seq.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                optim.zero_grad()
                src, src_lens = item.source
                trg, trg_lens = item.target
                output = mtseq2seq(src, src_lens, trg)
                output = output[1:].contiguous()
                output = output.view(-1, output.shape[-1])
                trg = trg.transpose(1, 0)
                trg = trg[1:].contiguous()
                trg = trg.view(-1)
                item_loss = F.cross_entropy(output, trg, ignore_index=TARGET.vocab.stoi['<pad>'])
                acc_loss += item_loss.item()
                item_loss.backward()
                clip_grad_norm_(mtseq2seq.parameters(), config.clip)
                optim.step()
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            print('epoch:{}, acc_loss:{}'.format(epoch, acc_loss))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        mtseq2seq.save()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = MTConfig.load(save_path)
        mtseq2seq = MTSeq2Seq(config)
        mtseq2seq .load()
        self._model = mtseq2seq
        self._source_vocab = config.source_word_vocab
        self._target_vocab = config.target_word_vocab

    def test(self, test_path):
        test_dataset = mt_tool.get_dataset(test_path)
        if not hasattr(SOURCE, 'vocab'):
            SOURCE.vocab = self._source_vocab
        if not hasattr(TARGET, 'vocab'):
            TARGET.vocab = self._source_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset, batch_size=DEFAULT_CONFIG['batch_size']):
        self._model.eval()
        dev_score_list = []
        dev_iter = mt_tool.get_iterator(dev_dataset, batch_size=batch_size)
        for dev_item in tqdm(dev_iter):
            src, src_lens = dev_item.source
            trg, trg_lens = dev_item.target
            item_score = mt_tool.get_score(self._model, src, src_lens, trg)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def predict(self, text: str, max_len=10):
        self._model.eval()
        text_list = eng_tokenize(text)
        text_list = [x.lower() for x in text_list]
        vec_text = torch.tensor([self._source_vocab.stoi[x] for x in text_list])
        vec_text = vec_text.reshape(1, -1).to(DEVICE)
        len_text = torch.tensor([len(vec_text)]).to(DEVICE)
        sos = torch.tensor([self._source_vocab.stoi['<sos>']]).to(DEVICE)
        output = self._model.predict(vec_text, len_text, sos, max_len).squeeze(1)
        soft_predict = torch.softmax(output, dim=1)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=1)
        predict_sentence = [self._target_vocab.itos[x] for x in predict_index]
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
