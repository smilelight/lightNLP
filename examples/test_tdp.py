import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.sp import TDP

tdp_model = TDP()

train_path = '../data/tdp/train.sample.txt'
dev_path = '../data/tdp/dev.txt'
vec_path = 'D:/Data/NLP/embedding/english/glove.6B.100d.txt'

tdp_model.train(train_path, dev_path=dev_path, vectors_path=vec_path,save_path='./tdp_saves',
                log_dir='E:/Test/tensorboard/')

tdp_model.load('./tdp_saves')
tdp_model.test(dev_path)
from pprint import pprint
pprint(tdp_model.predict('Investors who want to change the required timing should write their representatives '
                         'in Congress , he added . '))


