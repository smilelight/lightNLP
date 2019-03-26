import sys
sys.path.append('/home/lightsmile/Projects/MyGithub/lightNLP')

from lightnlp.kg.rl import RL

train_path = '/home/lightsmile/NLP/corpus/kg/baike/train.sample.csv'
dev_path = '/home/lightsmile/NLP/corpus/kg/baike/test.sample.csv'
model_type = 'TransE'

rl = RL()
# rl.train(train_path, model_type=model_type, dev_path=train_path, save_path='./rl_{}_saves'.format(model_type))

rl.load(save_path='./rl_{}_saves'.format(model_type), model_type=model_type)
# rl.test(train_path)

print(rl.predict_head(rel='外文名', tail='Compiler'))
print(rl.predict_rel(head='编译器', tail='Compiler'))
print(rl.predict_tail(head='编译器', rel='外文名'))
print(rl.predict(head='编译器', rel='外文名', tail='Compiler'))
