"""
逻辑回归
"""
from sklearn.linear_model import LogisticRegression as LR
from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator


class LogisticRegression:
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
    feature, label = FeatureExtractor.load_breast_cancer()
    train_feature, test_feature, train_label, test_label = FeatureExtractor.split_train_dataset(feature, label, 1/5, 8)
    lr = LogisticRegression()
    lr.train(train_feature, train_label)
    predictions = lr.predict(test_feature)
    print("predictions: ", predictions)
    classification_report = ModelEvaluator.get_classification_report(test_label, predictions,
                                                                     target_names=['Benign', 'Malignant'])
    print("classification_report: ", classification_report)


