from abc import ABCMeta, abstractclassmethod


class Tool(metaclass=ABCMeta):
    @abstractclassmethod
    def get_dataset(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def get_vocab(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def get_vectors(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def get_iterator(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def get_score(cls, *args, **kwargs):
        pass
