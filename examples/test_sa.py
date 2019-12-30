import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.tc import SA

sa_model = SA()

train_path = '../data/sa/train.sample.tsv'
dev_path = '../data/sa/dev.sample.tsv'
vec_path = 'D:/Data/NLP/embedding/word/sgns.zhihu.bigram-char'

sa_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./sa_saves',
               log_dir='E:/Test/tensorboard/')

# sa_model.load('./sa_saves')

from pprint import pprint

sa_model.test(train_path)

pprint(sa_model.predict('外观漂亮，安全性佳，动力够强，油耗够低。'))