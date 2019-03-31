import torch
from tqdm import tqdm
import torch.nn.functional as F

from ....utils.learning import adjust_learning_rate
from ....utils.log import logger
from ....base.module import Module

from ..model import Config
from .model import SkipGramNegativeSampling
from ..config import DEVICE, DEFAULT_CONFIG
from ..tool import skip_gram_tool, WORD
from ..utils.sampling import Sampling

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class SkipGramNegativeSamplingModule(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self.model_type = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = skip_gram_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = skip_gram_tool.get_dataset(dev_path)
            word_vocab = skip_gram_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab = skip_gram_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        train_iter = skip_gram_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, save_path=save_path, **kwargs)
        self.model_type = config.feature

        skip_gram = SkipGramNegativeSampling(config)
        # print(textcnn)
        self._model = skip_gram
        sampling = Sampling(word_vocab)
        optim = torch.optim.Adam(skip_gram.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            skip_gram.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                optim.zero_grad()
                neg = torch.tensor(sampling.sampling(config.neg_num *
                                                     item.context.size(0))).view(item.context.size(0), 1,
                                                                                 config.neg_num).to(DEVICE)
                item_loss = self._model.loss(item.target, item.context, neg)
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))

        config.save()
        skip_gram.save()

    def predict(self, target: str, topk=3):
        self._model.eval()

        vec_target = torch.tensor([self._word_vocab.stoi[target]])
        vec_target = vec_target.reshape(1, -1).to(DEVICE)
        vec_predict = self._model(vec_target)
        vec_score = F.softmax(vec_predict, dim=0)
        vec_prob, vec_index = torch.topk(vec_score, topk, dim=0)
        vec_prob = vec_prob.cpu().tolist()
        vec_index = [self._word_vocab.itos[i] for i in vec_index]

        return list(zip(vec_index, vec_prob))

    def evaluate(self, target, context):
        self._model.eval()

        vec_target = torch.tensor([self._word_vocab.stoi[target]])
        vec_target = vec_target.reshape(1, -1).to(DEVICE)
        vec_context = torch.tensor([self._word_vocab.stoi[context]])
        vec_context = vec_context.reshape(1, -1).to(DEVICE)
        score = self._model(vec_target, vec_context).item()

        return score

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        skip_gram = SkipGramNegativeSampling(config)
        skip_gram.load()
        self._model = skip_gram
        self._word_vocab = config.word_vocab
        self.model_type = config.feature
        self._check_vocab()

    def save_embeddings(self, save_path: str, save_mode='word2vec'):
        embeddings = self._model.word_embeddings.weight.data.cpu().numpy()
        with open(save_path, 'w', encoding='utf8') as f:
            if save_mode == 'word2vec':
                f.write('{} {}\n'.format(self._model.vocabulary_size, self._model.embedding_dimension))
            elif save_mode == 'glove':
                pass
            for word, word_id in self._word_vocab.stoi.items():
                if word == ' ':
                    continue
                word_embedding = embeddings[word_id]
                word_embedding = ' '.join(map(lambda x: str(x), word_embedding))
                f.write('{} {}\n'.format(word, word_embedding))
            logger.info('succeed saving embeddings data to {}'.format(save_path))

    def test(self, test_path):
        self._model.eval()
        test_dataset = skip_gram_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset, batch_size=DEFAULT_CONFIG['batch_size']):
        self._model.eval()
        dev_score_list = []
        dev_iter = skip_gram_tool.get_iterator(dev_dataset, batch_size=batch_size)
        for dev_item in tqdm(dev_iter):
            item_score = self._model(dev_item.target, dev_item.context).squeeze().cpu().tolist()
            dev_score_list.extend(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def _check_vocab(self):
        if not hasattr(WORD, 'vocab'):
            WORD.vocab = self._word_vocab
