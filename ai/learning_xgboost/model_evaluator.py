"""
模型评估器
"""
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


class ModelEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def get_accuracy(test_label, predictions):
        """
        获取准确率
        :param test_label:
        :param predictions:
        :return:
        """
        correct = 0
        for x in range(len(test_label)):
            if test_label[x] == predictions[x]:
                correct += 1
        return correct / float(len(test_label)) * 100.0

    @staticmethod
    def get_mean_squared_error(test_label, predictions):
        return mean_squared_error(test_label, predictions)

    @staticmethod
    def get_classification_report(test_label, predictions, target_names):
        return classification_report(test_label, predictions, target_names=target_names)