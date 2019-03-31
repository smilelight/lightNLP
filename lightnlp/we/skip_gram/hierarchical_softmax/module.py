import torch
from tqdm import tqdm
from typing import List

from ....utils.learning import adjust_learning_rate
from ....utils.log import logger
from ....base.module import Module

from ..model import Config
from .model import SkipGramHierarchicalSoftmax
from ..config import DEVICE, DEFAULT_CONFIG
from ..tool import skip_gram_tool, WORD
from ..utils.huffman_tree import HuffmanTree

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class SkipGramHierarchicalSoftmaxModule(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self.model_type = None
        self.huffman_tree = None
        self.huffman_pos_path = None
        self.huffman_neg_path = None

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

        skip_gram = SkipGramHierarchicalSoftmax(config)

        self._model = skip_gram
        word_id_frequency_dict = {self._word_vocab.stoi[s]: self._word_vocab.freqs[s] for s in self._word_vocab.stoi}
        self.huffman_tree = HuffmanTree(word_id_frequency_dict)
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        optim = torch.optim.SparseAdam(skip_gram.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            skip_gram.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                optim.zero_grad()
                pos_pairs = []
                neg_pairs = []
                for i in range(item.batch_size):
                    pos_path = self.huffman_pos_path[item.context[i]]
                    neg_path = self.huffman_neg_path[item.context[i]]
                    pos_pairs.extend(zip([item.target[i]] * len(pos_path), pos_path))
                    neg_pairs.extend(zip([item.target[i]] * len(neg_path), neg_path))
                pos_context_vec = torch.cat(tuple(pair[0].view(1, -1) for pair in pos_pairs), dim=0).to(DEVICE)
                pos_path_vec = torch.tensor([[pair[1]] for pair in pos_pairs]).to(DEVICE)
                neg_context_vec = torch.cat(tuple(pair[0].view(1, -1) for pair in neg_pairs), dim=0).to(DEVICE)
                neg_path_vec = torch.tensor([[pair[1]] for pair in neg_pairs]).to(DEVICE)

                item_loss = self._model.loss(pos_context_vec, pos_path_vec, neg_context_vec, neg_path_vec)
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

        random_predict = torch.rand(len(self._word_vocab.itos)).to(DEVICE)
        vec_prob, vec_index = torch.topk(random_predict, topk, dim=0)
        vec_prob = vec_prob.cpu().tolist()
        vec_index = [self._word_vocab.itos[i] for i in vec_index]
        return list(zip(vec_index, vec_prob))

    def evaluate(self, context, target):
        self._model.eval()
        vec_context = torch.tensor([self._word_vocab.stoi[x] for x in context])
        vec_context = vec_context.to(DEVICE)
        pos_path = self.huffman_pos_path[self._word_vocab.stoi[target]]
        neg_path = self.huffman_neg_path[self._word_vocab.stoi[target]]
        pos_pairs = list(zip([vec_context] * len(pos_path), pos_path))
        neg_pairs = list(zip([vec_context] * len(neg_path), neg_path))
        pos_context_vec = torch.cat(tuple(pair[0].view(1, -1) for pair in pos_pairs), dim=0).to(DEVICE)
        pos_path_vec = torch.tensor([[pair[1]] for pair in pos_pairs]).to(DEVICE)
        neg_context_vec = torch.cat(tuple(pair[0].view(1, -1) for pair in neg_pairs), dim=0).to(DEVICE)
        neg_path_vec = torch.tensor([[pair[1]] for pair in neg_pairs]).to(DEVICE)

        item_score = self._model(pos_context_vec, pos_path_vec, neg_context_vec, neg_path_vec)
        return item_score

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        skip_gram = SkipGramHierarchicalSoftmax(config)
        skip_gram.load()
        self._model = skip_gram
        self._word_vocab = config.word_vocab
        self.model_type = config.feature
        self._check_vocab()
        word_id_frequency_dict = {self._word_vocab.stoi[s]: self._word_vocab.freqs[s] for s in self._word_vocab.stoi}
        self.huffman_tree = HuffmanTree(word_id_frequency_dict)
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()

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
            pos_pairs = []
            neg_pairs = []
            for i in range(dev_item.batch_size):
                pos_path = self.huffman_pos_path[dev_item.context[i]]
                neg_path = self.huffman_neg_path[dev_item.context[i]]
                pos_pairs.extend(zip([dev_item.target[i]] * len(pos_path), pos_path))
                neg_pairs.extend(zip([dev_item.target[i]] * len(neg_path), neg_path))
            pos_context_vec = torch.cat(tuple(pair[0].view(1, -1) for pair in pos_pairs), dim=0).to(DEVICE)
            pos_path_vec = torch.tensor([[pair[1]] for pair in pos_pairs]).to(DEVICE)
            neg_context_vec = torch.cat(tuple(pair[0].view(1, -1) for pair in neg_pairs), dim=0).to(DEVICE)
            neg_path_vec = torch.tensor([[pair[1]] for pair in neg_pairs]).to(DEVICE)

            item_score = self._model(pos_context_vec, pos_path_vec, neg_context_vec, neg_path_vec)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def _check_vocab(self):
        if not hasattr(WORD, 'vocab'):
            WORD.vocab = self._word_vocab
