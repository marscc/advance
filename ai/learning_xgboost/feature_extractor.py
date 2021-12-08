"""
特征抽取
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def load(path, sep, names, header):
        return pd.read_csv(path, sep=sep, names=names, header=header)

    @staticmethod
    def load_iris():
        iris = datasets.load_iris()
        return iris.data, iris.target, iris.feature_names

    @staticmethod
    def load_boston():
        boston = datasets.load_boston()
        return boston.data, boston.target

    @staticmethod
    def load_breast_cancer():
        cancer = datasets.load_breast_cancer()
        return cancer.data, cancer.target

    @staticmethod
    def split_train_dataset(dataset, label, ratio, random_state):
        train_feature, test_feature, train_label, test_label = train_test_split(dataset, label, test_size=ratio,
                                                                                random_state=random_state)
        return train_feature, test_feature, train_label, test_label

    @staticmethod
    def split_train_dataset1(dataset, mode, ratio):
        # 换出法
        if "hold-out" == mode:
            # 1.生成一个len(iris_data)长度的一维向量，向量里的元素服从0~1分布
            # 2.判断向量中每个元素和0.8的大小，生成一个值为true或false的掩码向量
            msk = np.random.rand(len(dataset)) < ratio
            # 3.根据掩码向量分别生成训练集和测试集
            train_data_origin = dataset[msk]
            test_data_origin = dataset[~msk]
            # 4.重置索引
            train_data = train_data_origin.reset_index(drop=True)
            test_data = test_data_origin.reset_index(drop=True)
            # 5.训练集label和测试集label
            train_label = train_data['class']
            test_label = test_data['class']
            # 6.训练集和测试集feature
            train_feature = train_data.drop('class', axis=1)
            test_feature = test_data.drop('class', axis=1)
            print(train_feature)
            train_norm = FeatureExtractor.normalize(train_feature)
            test_norm = FeatureExtractor.normalize(test_feature)
            return train_norm, test_norm, train_label, test_label

    @staticmethod
    def normalize(feature):
        """
        归一化
        :param feature:
        :return:
        """
        return (feature - feature.min()) / (feature.max() - feature.min())
