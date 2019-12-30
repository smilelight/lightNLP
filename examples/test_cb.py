import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.tg import CB

cb_model = CB()

train_path = '../data/cb/chat.train.sample.tsv'
dev_path = '../data/cb/chat.test.sample.tsv'
vec_path = 'D:/Data/NLP/embedding/word/sgns.zhihu.bigram-char'

# cb_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./cb_saves',
#                log_dir='E:/Test/tensorboard/')

cb_model.load('./cb_saves')

# cb_model.test(train_path)

# print(cb_model.predict('我还喜欢她,怎么办'))
# print(cb_model.predict('怎么了'))
# print(cb_model.predict('开心一点'))

cb_model.deploy()
