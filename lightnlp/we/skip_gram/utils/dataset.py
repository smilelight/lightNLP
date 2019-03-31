from torchtext.data import Dataset, Example
from torchtext.data import Field

import jieba

CONTEXT = Field(tokenize=lambda x: [x], batch_first=True)
TARGET = Field(tokenize=lambda x: [x], batch_first=True)
Fields = [
            ('context', CONTEXT),
            ('target', TARGET)
        ]


def default_tokenize(sentence):
    return list(jieba.cut(sentence))


class SkipGramDataset(Dataset):

    def __init__(self, path, fields, window_size=3, tokenize=default_tokenize, encoding="utf-8", **kwargs):
        examples = []
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                words = tokenize(line.strip())
                if len(words) < window_size + 1:
                    continue
                for i in range(len(words)):
                    contexts = words[max(0, i - window_size):i] + \
                               words[min(i+1, len(words)):min(len(words), i + window_size) + 1]
                    for context in contexts:
                        examples.append(Example.fromlist((context, words[i]), fields))
        super(SkipGramDataset, self).__init__(examples, fields, **kwargs)


if __name__ == '__main__':
    test_path = '/home/lightsmile/NLP/corpus/novel/test.txt'
    dataset = SkipGramDataset(test_path, Fields)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0].context)
    print(dataset[0].target)

    TARGET.build_vocab(dataset)

    from sampling import Sampling

    samp = Sampling(TARGET.vocab)

    print(samp.sampling(3))
