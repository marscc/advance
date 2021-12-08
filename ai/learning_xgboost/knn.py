"""
K近邻
"""
from sklearn import neighbors
from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator


class KNN:
    def __init__(self):
        self.knn = neighbors.KNeighborsClassifier(n_neighbors=3)

    def train(self, train_features, train_labels):
        """
        模型训练
        :param train_features:
        :param train_labels:
        :return:
        """
        self.knn.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.knn.predict(test_features)


if __name__ == '__main__':
    ds = FeatureExtractor.load(path='/Users/liuyang/Mars/project/data_set/iris.csv',
                               sep=',',
                               names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
                               header=1)
    train_norm, test_norm, train_label, test_label = FeatureExtractor.split_train_dataset1(ds, "hold-out", 0.8)
    knn = KNN()
    knn.train(train_features=train_norm, train_labels=train_label)
    pred_res = knn.predict(test_features=test_norm)
    print("prediction res: ", pred_res)
    accuracy = ModelEvaluator.get_accuracy(test_label, pred_res)
    print("accuracy: ", accuracy)

