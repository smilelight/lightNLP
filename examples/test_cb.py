from lightnlp.tg import CB

cb_model = CB()

train_path = '/home/lightsmile/NLP/corpus/chatbot/chat.train.sample.tsv'
dev_path = '/home/lightsmile/NLP/corpus/chatbot/chat.test.sample.tsv'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'

# cb_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./cb_saves')

cb_model.load('./cb_saves')

# cb_model.test(train_path)

print(cb_model.predict('我还喜欢她,怎么办'))
print(cb_model.predict('怎么了'))
print(cb_model.predict('开心一点'))
