from lightnlp.sl import SA

sa_model = SA()

train_path = '/home/lightsmile/Projects/NLP/chinese_text_cnn/data/train.sample.tsv'
dev_path = '/home/lightsmile/Projects/NLP/chinese_text_cnn/data/dev.sample.tsv'
vec_path = '/home/lightsmile/Downloads/1410356697_浅笑哥fight/自然语言处理/词向量/sgns.zhihu.bigram-char'

# sa_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./sa_saves')

sa_model.load()

from pprint import pprint

sa_model.test(train_path)

# pprint(sa_model.predict('另一个很酷的事情是，通过框架我们可以停止并在稍后恢复训练。'))