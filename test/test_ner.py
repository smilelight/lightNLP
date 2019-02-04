from lightnlp.sl import NER

ner_model = NER()

train_path = '/home/lightsmile/Download/ner/test.sample.txt'
dev_path = '/home/lightsmile/Download/ner/train.sample.txt'
vec_path = '/home/lightsmile/Projects/NLP/ChineseEmbedding/model/token_vec_300.bin'

ner_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./ner_saves')

# ner_model.load()

# from pprint import pprint

# ner_model.test(train_path)

# pprint(ner_model.predict('另一个很酷的事情是，通过框架我们可以停止并在稍后恢复训练。'))
