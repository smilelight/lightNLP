import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])

from lightnlp.we import CBOWHierarchicalSoftmaxModule, CBOWNegativeSamplingModule, CBOWBaseModule

# cbow_model = CBOWHierarchicalSoftmaxModule()
cbow_model = CBOWBaseModule()
# cbow_model = CBOWNegativeSamplingModule()

train_path = '../data/novel/test.txt'
dev_path = '../data/novel/test.txt'

# cbow_model.train(train_path, dev_path=dev_path, save_path='./cbow_saves', log_dir='E:/Test/tensorboard/')

cbow_model.load('./cbow_saves')
cbow_model.deploy()


#
# cbow_model.test(dev_path)

# test_context = ['族长', '是', '的', '父亲']
# print(cbow_model.predict(test_context))
# print(cbow_model.evaluate(test_context, '他'))
# print(cbow_model.evaluate(test_context, '提防'))

# cbow_model.save_embeddings('./cbow_saves/cbow_hs.bin')
# cbow_model.save_embeddings('./cbow_saves/cbow_base.bin')
# cbow_model.save_embeddings('./cbow_saves/cbow_ns.bin')
