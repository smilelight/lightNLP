import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.tg import LM

lm_model = LM()

train_path = '../data/lm/lm_test.txt'
dev_path = '../data/lm/lm_test.txt'
vec_path = 'D:/Data/NLP/embedding/char/token_vec_300.bin'

# lm_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./lm_saves',
#                log_dir='E:/Test/tensorboard/')

lm_model.load('./lm_saves')
lm_model.deploy()
# lm_model.test(train_path)

# print(lm_model.next_word('要不是', '萧'))
#
# print(lm_model.generate_sentence('少年面无表情，唇角有着一抹自嘲'))
#
# print(lm_model.next_word_topk('少年面无表情，唇角'))
#
# print(lm_model.sentence_score('少年面无表情，唇角有着一抹自嘲'))
