from torchtext.data import Dataset, Example
from .preprocess import handle_line


class REDataset(Dataset):
    """Defines a Dataset of relation extraction format.
    eg:
    钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
    元武	元华	unknown	于师傅在一次京剧表演中，选了元龙（洪金宝）、元楼（元奎）、元彪、成龙、元华、元武、元泰7人担任七小福的主角。
    """

    def __init__(self, path, fields, encoding="utf-8", **kwargs):
        examples = []
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                chunks = line.split()
                entity_1, entity_2, relation, sentence = tuple(chunks)
                sentence_list = handle_line(entity_1, entity_2, sentence)

                examples.append(Example.fromlist((sentence_list, relation), fields))
        super(REDataset, self).__init__(examples, fields, **kwargs)

