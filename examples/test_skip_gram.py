from lightnlp.we import SkipGramBaseModule, SkipGramNegativeSamplingModule, SkipGramHierarchicalSoftmaxModule

# skip_gram_model = SkipGramHierarchicalSoftmaxModule()
skip_gram_model = SkipGramNegativeSamplingModule()
# skip_gram_model = SkipGramBaseModule()

train_path = 'D:/Data/NLP/corpus/novel/test.txt'
dev_path = 'D:/Data/NLP/corpus/novel/test.txt'

skip_gram_model.train(train_path, dev_path=dev_path, save_path='./skip_gram_saves')

skip_gram_model.load('./skip_gram_saves')
#
# skip_gram_model.test(dev_path)

test_target = '族长'
# print(skip_gram_model.predict(test_target))
print(skip_gram_model.evaluate(test_target, '他'))
print(skip_gram_model.evaluate(test_target, '提防'))

# skip_gram_model.save_embeddings('./skip_gram_saves/skip_gram_hs.bin')
# skip_gram_model.save_embeddings('./skip_gram_saves/skip_gram_base.bin')
skip_gram_model.save_embeddings('./skip_gram_saves/skip_gram_ns.bin')
