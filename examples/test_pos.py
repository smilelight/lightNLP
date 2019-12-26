import sys
sys.path.append('E:/Projects/myProjects/lightNLP')

from lightnlp.sl import POS

pos_model = POS()

train_path = 'D:/Data/NLP/corpus/pos/train.sample.txt'
dev_path = 'D:/Data/NLP/corpus/pos/test.sample.txt'
vec_path = 'D:/Data/NLP/embedding/char/token_vec_300.bin'

# pos_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./pos_saves')

pos_model.load('./pos_saves')

# pos_model.test(dev_path)

print(pos_model.predict('向全国各族人民致以诚挚的问候！'))