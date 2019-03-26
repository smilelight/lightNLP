import sys
sys.path.append('/home/lightsmile/Projects/MyGithub/lightNLP')

from lightnlp.tg import LM

lm_model = LM()

train_path = '/home/lightsmile/NLP/corpus/language_model/lm_test.txt'
dev_path = '/home/lightsmile/NLP/corpus/language_model/lm_test.txt'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

# lm_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./lm_saves')

lm_model.load('./lm_saves')

# lm_model.test(train_path)

print(lm_model.next_word('要不是', '萧'))

print(lm_model.generate_sentence('少年面无表情，唇角有着一抹自嘲'))

print(lm_model.next_word_topk('少年面无表情，唇角'))

print(lm_model.sentence_score('少年面无表情，唇角有着一抹自嘲'))
