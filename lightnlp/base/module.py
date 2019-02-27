from abc import ABCMeta, abstractclassmethod


class Module(metaclass=ABCMeta):
    @abstractclassmethod
    def train(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def load(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def _validate(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def test(cls, *args, **kwargs):
        pass
