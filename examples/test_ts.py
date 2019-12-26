from lightnlp.tg import TS

ts_model = TS()

train_path = 'D:/Data/NLP/corpus/ts/train.sample.tsv'
dev_path = 'D:/Data/NLP/corpus/ts/test.sample.tsv'
vec_path = 'D:/Data/NLP/embedding/word/sgns.zhihu.bigram-char'

# ts_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./ts_saves')

ts_model.load('./ts_saves')

ts_model.test(train_path)

test_str = """
            近日，因天气太热，安徽一老太在买肉路上突然眼前一黑，摔倒在地。她怕别人不扶她，连忙说"快扶我起来，我不讹你，地上太热我要熟了！"这一喊周围人都笑了，老人随后被扶到路边休息。(颍州晚报)[话筒]最近老人尽量避免出门!
            """

print(ts_model.predict(test_str))
