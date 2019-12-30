import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.sl import NER

ner_model = NER()

train_path = '../data/ner/train.sample.txt'
dev_path = '../data/ner/test.sample.txt'
vec_path = 'D:/Data/NLP/embedding/char/token_vec_300.bin'

ner_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./ner_saves',
                log_dir='E:/Test/tensorboard/')

ner_model.load('./ner_saves')

from pprint import pprint

ner_model.test(train_path)

pprint(ner_model.predict('另一个很酷的事情是，通过框架我们可以停止并在稍后恢复训练。'))
