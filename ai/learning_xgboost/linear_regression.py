"""
线性回归
"""

from sklearn.linear_model import LinearRegression as LR
from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator


class LinearRegression:
    def __init__(self):
        self.lr = LR()

    def train(self, train_features, train_labels):
        """
        模型训练
        :param train_features:
        :param train_labels:
        :return:
        """
        self.lr.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.lr.predict(test_features)


if __name__ == "__main__":
    data, target = FeatureExtractor.load_boston()
    train_feature, test_feature, train_label, test_label = FeatureExtractor.split_train_dataset1(data, target, 1 / 5, 8)
    lr = LinearRegression()
    lr.train(train_feature, train_label)
    predictions = lr.predict(test_feature)
    print("predictions:", predictions)
    mse = ModelEvaluator.get_mean_squared_error(test_label, predictions)
    print("mse:", mse)
