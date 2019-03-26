import sys
sys.path.append('/home/lightsmile/Projects/MyGithub/lightNLP')

from lightnlp.sl import SRL

srl_model = SRL()

train_path = '/home/lightsmile/NLP/corpus/srl/train.sample.tsv'
dev_path = '/home/lightsmile/NLP/corpus/srl/test.sample.tsv'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'


# srl_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./srl_saves')

srl_model.load('./srl_saves')

# srl_model.test(dev_path)

word_list = ['代表', '朝方', '对', '中国', '党政', '领导人', '和', '人民', '哀悼', '金日成', '主席', '逝世', '表示', '深切', '谢意', '。']
pos_list = ['VV', 'NN', 'P', 'NR', 'NN', 'NN', 'CC', 'NN', 'VV', 'NR', 'NN', 'VV', 'VV', 'JJ', 'NN', 'PU']
rel_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(srl_model.predict(word_list, pos_list, rel_list))

