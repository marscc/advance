"""
简单GBDT模型,实现的功能如下：
1. 拟合残差
todo: 剪枝
"""
from cart import Cart
import numpy as np


class SimpleGBDT:
    def __init__(self, max_tree_num):
        self.trees = []
        self.max_tree_num = max_tree_num

    def fit(self, features, targets):
        """
        模型训练
        :param features: 特征
        :param targets: label
        :return:
        """
        # 残差
        residual = targets
        for idx in range(self.max_tree_num):
            model = Cart()
            self.trees.append(model)
            model.fit(features, residual)
            y = model.predict(features)
            residual = residual - y

    def predict(self, features):
        """
        模型预测
        :param features: 特征
        :return: 预测值
        """
        y = np.zeros(features.shape[0])
        for model in self.trees:
            prediction = np.array(model.predict(features))
            y += prediction
        return y

