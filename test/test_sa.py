from lightnlp.tc import SA

sa_model = SA()

train_path = '/home/lightsmile/Projects/NLP/chinese_text_cnn/data/train.sample.tsv'
dev_path = '/home/lightsmile/Projects/NLP/chinese_text_cnn/data/dev.sample.tsv'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'

# sa_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./sa_saves')

sa_model.load('./sa_saves')

from pprint import pprint

sa_model.test(train_path)

pprint(sa_model.predict('外观漂亮，安全性佳，动力够强，油耗够低。'))