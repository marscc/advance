"""
分类模型的损失函数基类
"""
import abc


class ClassificationLossFunction(metaclass=abc.ABCMeta):
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """
        计算残差
        :param dataset:
        :param subset:
        :param f:
        :return:
        """

    
