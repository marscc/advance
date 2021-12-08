"""
CART（基于Sklearn）
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CART:
    def __init__(self):
        self.tree = DecisionTreeRegressor(max_depth=4)

    def train(self, train_features, train_labels):
        """
        模型训练
        :param train_features:
        :param train_labels:
        :return:
        """
        self.tree.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.tree.predict(test_features)

    def get_model(self):
        return self.tree


if __name__ == '__main__':
    X = np.arange(1, 11).reshape(-1, 1)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    cart = CART()
    cart.train(X, y)
    from six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO()
    export_graphviz(cart.get_model(), out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, precision=2)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.write_png("cart.png"))