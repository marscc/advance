"""
构造决策树
步骤：
 1. 计算每个特征的信息增益
 2. 选取信息增益最大的特征作为当前分裂节点的特征
 3. 递归执行以上2步，直到分裂节点下的结果为同一类别（假设是分类问题）
"""
import math
import collections
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from feature_extractor import FeatureExtractor


class DecisionTree:
    def __init__(self):
        self.dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    def train(self, train_features, train_labels):
        """
        模型训练
        :param train_features:
        :param train_labels:
        :return:
        """
        self.dtree.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.dtree.predict(test_features)

    def get_model(self):
        return self.dtree

    @staticmethod
    def entropy(rows: list) -> float:
        """
        计算数组的熵
        :param rows: 数组
        :return: 信息熵
        """
        result = collections.Counter()
        result.update(rows)
        rows_len = len(rows)
        assert rows_len
        ent = 0.0
        for x in result.values():
            p = float(x) / rows_len
            ent -= p * math.log2(p)
        return ent

    @staticmethod
    def condition_entropy(future_list: list, result_list: list) -> float:
        """
        计算条件熵
        :param future_list: 列表，eg. ["beijing", "beijing", "beijing", "shanghai", "shanghai", "shanghai", "shanghai"]
        :param result_list: 结果列表，eg. [1, 0, 1, 0, 1, 1, 1]
        :return:
        """
        entropy_dict = collections.defaultdict(list)
        for future, value in zip(future_list, result_list):
            entropy_dict[future].append(value)
        ent = 0.0
        future_len = len(future_list)
        for x in entropy_dict.values():
            p = len(x) / future_len * DecisionTree.entropy(x)
            ent += p
        return ent

    @staticmethod
    def info_gain(future_list: list, result_list: list) -> float:
        """
        计算信息增益
        :param future_list:
        :param result_list:
        :return:
        """
        entropy = DecisionTree.entropy(result_list)
        condition_entropy = DecisionTree.condition_entropy(future_list, result_list)
        return entropy - condition_entropys


if __name__ == "__main__":
    # 计算熵
    a = [1, 0, 0, 1]
    print("entropy: ", DecisionTree.entropy(a))
    # 计算条件熵
    a = ["beijing", "beijing", "beijing", "shanghai", "shanghai", "shanghai", "shanghai"]
    b = [1, 0, 1, 0, 1, 1, 1]
    print("condition_entropy: ", DecisionTree.condition_entropy(a, b))
    print("info_gain: ", DecisionTree.info_gain(a, b))

    # ID3
    train_features, train_labels, train_feature_names = FeatureExtractor.load_iris()
    df = pd.DataFrame(train_features, columns=train_feature_names)
    dtree = DecisionTree()
    dtree.train(train_features, train_labels)
    from six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO()
    export_graphviz(dtree.get_model(), out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, precision=2)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.write_png("iris.png"))


