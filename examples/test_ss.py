import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.sr import SS

ss_model = SS()

train_path = '../data/ss/ss_train.tsv'
dev_path = '../data/ss/ss_dev.tsv'
vec_path = 'D:/Data/NLP/embedding/char/token_vec_300.bin'

ss_model.train(train_path, vectors_path=vec_path, dev_path=train_path, save_path='./ss_saves',
               log_dir='E:/Test/tensorboard/')


# ss_model.load('./ss_saves')
# ss_model.test(dev_path)

print(float(ss_model.predict('花呗收款收不了怎么办', '开通花呗收款为何不能用')))
print(float(ss_model.predict('花呗的安全没有验证成功', '花呗安全验证没通过怎么回事')))
print(float(ss_model.predict('花呗支付可以使用购物津贴吗', '使用购物津贴的费用可以用花呗吗')))
print(float(ss_model.predict('花呗更改绑定银行卡', '如何更换花呗绑定银行卡')))