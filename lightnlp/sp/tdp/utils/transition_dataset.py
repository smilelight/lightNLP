from torchtext.data import Dataset, Example


class TransitionDataset(Dataset):
    """Defines a Dataset of transition-based denpendency parsing format.
    eg:
    The bill intends ||| SHIFT SHIFT REDUCE_L SHIFT REDUCE_L
    The bill intends ||| SHIFT SHIFT REDUCE_L SHIFT REDUCE_L
    """

    def __init__(self, path, fields, encoding="utf-8", separator=' ||| ', **kwargs):
        examples = []
        with open(path, "r", encoding=encoding) as f:
            
            for inst in f:
                sentence, actions = inst.split(separator)

                # Make sure there is no leading/trailing whitespace
                sentence = sentence.strip().split()
                actions = actions.strip().split()

                examples.append(Example.fromlist((sentence, actions), fields))
        super(TransitionDataset, self).__init__(examples, fields, **kwargs)

