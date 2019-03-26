import sys
sys.path.append('/home/lightsmile/Projects/MyGithub/lightNLP')

from lightnlp.sp import GDP

gdp_model = GDP()

train_path = '/home/lightsmile/NLP/corpus/dependency_parse/gdp/THU/train.sample.conll'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'


# gdp_model.train(train_path, dev_path=train_path, vectors_path=vec_path, save_path='./gdp_saves')

gdp_model.load('./gdp_saves')
# gdp_model.test(train_path)
word_list = ['最高', '人民', '检察院', '检察长', '张思卿']
pos_list = ['nt', 'nt', 'nt', 'n', 'nr']
heads, rels = gdp_model.predict(word_list, pos_list)
print(heads)
print(rels)