from lightnlp.tg import MT

mt_model = MT()

train_path = '/home/lightsmile/NLP/corpus/translation/mt.train.sample.tsv'
dev_path = '/home/lightsmile/NLP/corpus/translation/mt.test.sample.tsv'
source_vec_path = '/home/lightsmile/NLP/embedding/english/glove.6B.100d.txt'
target_vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'

# mt_model.train(train_path, source_vectors_path=source_vec_path, target_vectors_path=target_vec_path,
#                dev_path=train_path, save_path='./mt_saves')

mt_model.load('./mt_saves')

mt_model.test(train_path)

print(mt_model.predict('Hello!'))
print(mt_model.predict('Wait!'))
