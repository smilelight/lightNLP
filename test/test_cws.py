import sys
sys.path.append('/home/lightsmile/Projects/MyGithub/lightNLP')

from lightnlp.sl import CWS

cws_model = CWS()

train_path = '/home/lightsmile/NLP/corpus/cws/train.sample.txt'
dev_path = '/home/lightsmile/NLP/corpus/cws/test.sample.txt'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

# cws_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./cws_saves')

cws_model.load('./cws_saves')


# cws_model.test(dev_path)

print(cws_model.predict('抗日战争时期，胡老在与侵华日军交战中四次负伤，是一位不折不扣的抗战老英雄'))