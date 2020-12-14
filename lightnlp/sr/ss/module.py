import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import flask
from flask import Flask, request

from ...utils.deploy import get_free_tcp_port
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

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None,
              **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = ss_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ss_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = ss_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = ss_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._label_vocab = tag_vocab
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        train_iter = ss_tool.get_iterator(train_dataset, batch_size=config.batch_size)
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
            writer.add_scalar('ss_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('ss_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
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

    def deploy(self, route_path="/ss", host="localhost", port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            texta = request.args.get('texta', '')
            textb = request.args.get('textb', '')
            result = self.predict(texta, textb)
            return flask.jsonify({
                    'state': 'OK',
                    'result': {
                        'prob': result
                        }
                })
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)
