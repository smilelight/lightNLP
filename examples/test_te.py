import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.sr import TE

te_model = TE()

train_path = '../data/te/te_train.tsv'
dev_path = '../data/te_dev.tsv'
vec_path = 'D:/Data/NLP/embedding/char/token_vec_300.bin'

te_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./te_saves',
               log_dir='E:/Test/tensorboard/')

te_model.load('./te_saves')
# te_model.test(dev_path)

print(te_model.predict('一个小男孩在秋千上玩。', '小男孩玩秋千'))
print(te_model.predict('两个年轻人用泡沫塑料杯子喝酒时做鬼脸。', '两个人在跳千斤顶。'))