import sys
sys.path.append('E:/Projects/myProjects/lightNLP')

from lightnlp.sp import TDP

tdp_model = TDP()

train_path = 'D:/Data/NLP/corpus/tdp/train.sample.txt'
dev_path = 'D:/Data/NLP/corpus/tdp/dev.txt'
vec_path = 'D:/Data/NLP/embedding/english/glove.6B.100d.txt'

tdp_model.train(train_path, dev_path=dev_path, vectors_path=vec_path,save_path='./tdp_saves')

tdp_model.load('./tdp_saves')
tdp_model.test(dev_path)
from pprint import pprint
pprint(tdp_model.predict('Investors who want to change the required timing should write their representatives '
                         'in Congress , he added . '))


