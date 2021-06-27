import abc
from abc import ABCMeta


class Experiment(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, '_train') and
            callable(subclass._train) and
            hasattr(subclass, 'run') and
            callable(subclass.run) and
            hasattr(subclass, '_test') and
            callable(subclass._test) or
            NotImplemented
        )

    @abc.abstractmethod
    def _train(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def _test(self):
        pass
