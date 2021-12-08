"""
CART树的完整实现：
1. 缺失值如何处理？
2. 剪枝
"""


class TreeNode:
    """
    Cart树的内部节点
    """

    def __init__(self, split_feature_id, split_feature_value):
        self.split_feature_id = split_feature_id
        self.split_feature_value = split_feature_value
        self.left_child = None
        self.right_child = None
        self.leaf_node = None

    def set_leaf_node(self, leaf_node):
        self.leaf_node = leaf_node

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    def get_split_feature_id(self):
        return self.split_feature_id

    def get_split_feature_value(self):
        return self.split_feature_value

    def get_leaf_node(self):
        return self.leaf_node

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def info(self):
        pass


class LeafNode:
    """
    CART树的叶子节点
    """

    def __init__(self, id_set, predict_value):
        self.id_set = id_set
        self.predict_value = predict_value

    def get_id_set(self):
        return self.predict_value

    def get_predict_value(self):
        return self.predict_value

    def set_predict_value(self, predict_value):
        self.predict_value = predict_value

    def info(self):
        return "{LeafNode:" + str(self.predict_value) + "}"


class LossUtil:
    def __init__(self):
        pass

    @staticmethod
    def mse(features, targets, candidate_sample_ids, feature_id, sample_id):
        """
        计算左右子树最小均方误差的和
        :param features: 特征集，m*n介矩阵
        :param targets: label向量
        :param candidate_sample_ids: 候选样本ids（随着不停的划分，ids会越来越少）
        :param feature_id: 样本ID
        :param sample_id: 特征ID
        :return:
        """
        # 分割值
        split_value = features[sample_id, feature_id]
        # 左边的样本ids, 右边的样本ids
        left_sample_ids, right_sample_ids = [], []
        # 左右空间各自的label和，为了求均值
        left_y_sum, right_y_sum = 0, 0
        for idx in candidate_sample_ids:
            if features[idx, feature_id] <= split_value:
                left_sample_ids.append(idx)
                left_y_sum += targets[idx]
            else:
                right_sample_ids.append(idx)
                right_y_sum += targets[idx]
        # 求均值
        left_y_mean = left_y_sum / len(left_sample_ids) if len(left_sample_ids) > 0 else 0
        right_y_mean = right_y_sum / len(right_sample_ids) if len(right_sample_ids) > 0 else 0
        # 求mse
        left_mse = 0.0
        right_mse = 0.0
        for idx in left_sample_ids:
            left_mse += (targets[idx] - left_y_mean) ** 2
        for idx in right_sample_ids:
            right_mse += (targets[idx] - right_y_mean) ** 2
        return left_mse + right_mse


class Cart:
    """
    CART回归树，实现：
      1. fit
      2. predict
    """

    def __init__(self):
        self.feature_num = None
        self.tree = None
        self.tree_size = 0
        self.max_depth = 10

    def fit(self, features, targets):
        """
        拟合
        :param features: 特征
        :param targets: label
        :return: None
        """
        self.feature_num = len(features[0])
        root = TreeNode(-1, None)
        sample_ids = range(len(features))
        depth = 0
        self.__build_tree(features, targets, sample_ids, root, depth, True)
        self.tree = root

    def predict(self, features):
        """
        预测
        :param features: 特征
        :return: 预测值
        """
        prediction_list = []
        for sample_id in range(features.shape[0]):
            prediction = self.__search(features[sample_id])
            prediction_list.append(prediction)
        return prediction_list

    def __build_tree(self, features, targets, candidate_sample_ids, parent_tree_node: TreeNode, depth, is_left_child):
        """
        构建决策树
        :param features: 特征集
        :param targets: label向量
        :param candidate_sample_ids: 候选样本id集合
        :param parent_tree_node: 当前父节点
        :param depth: 树的深度
        :param is_left_child: 是否是左孩子
        :return:
        """
        # 递归终止条件
        if len(candidate_sample_ids) == 0:
            # 如果没有样本集合了，直接返回
            return
        elif len(candidate_sample_ids) == 1 or depth > self.max_depth:
            # 到了只有一个样本，或者划分到当前树的深度已经超过阈值，添加叶子节点
            predict_value = sum([targets[idx] for idx in candidate_sample_ids]) / len(candidate_sample_ids)
            leaf_node = LeafNode(candidate_sample_ids, predict_value)
            parent_tree_node.set_leaf_node(leaf_node)
            return
        else:
            # 否则，递归的去划分特征空间
            best_split_feature_id, best_split_feature_value, left_sample_ids, right_sample_ids = self.__get_best_split(
                features, targets, candidate_sample_ids)
            tree_node = TreeNode(best_split_feature_id, best_split_feature_value)
            if is_left_child:
                parent_tree_node.set_left_child(tree_node)
            else:
                parent_tree_node.set_right_child(tree_node)
            self.tree_size += 1
            depth += 1
            # 开始递归左右子树
            self.__build_tree(features, targets, left_sample_ids, tree_node, depth, True)
            self.__build_tree(features, targets, right_sample_ids, tree_node, depth, False)

    def __get_best_split(self, features, targets, candidate_sample_ids):
        """
        获取最佳划分
        :param features: 特征集合
        :param targets: labels
        :param candidate_sample_ids: 候选样本ids
        :return:
            best_split_feature_id: 最佳划分的特征id
            best_split_feature_value: 最佳划分的特征值
            left_sample_ids: 划分到左空间的样本id
            right_sample_ids: 划分到右空间的样本id
        """
        feature_ids = range(self.feature_num)
        # 最小的那个MSE
        best_mse = -1
        best_split_feature_id = None
        best_split_feature_value = None
        # 遍历所有特征（相当于书中提到的j）
        for feature_id in feature_ids:
            # 遍历某个特征下的所有样本值（相当于书中提到的s）
            for sample_id in candidate_sample_ids:
                current_mse = LossUtil.mse(features, targets, candidate_sample_ids, feature_id, sample_id)
                if best_mse == -1 or current_mse < best_mse:
                    best_mse = current_mse
                    best_split_feature_id = feature_id
                    best_split_feature_value = features[sample_id, feature_id]
        left_sample_ids = []
        right_sample_ids = []
        for idx in candidate_sample_ids:
            if features[idx, best_split_feature_id] <= best_split_feature_value:
                left_sample_ids.append(idx)
            else:
                right_sample_ids.append(idx)
        return best_split_feature_id, best_split_feature_value, left_sample_ids, right_sample_ids

    def __prune(self):
        pass

    def __search(self, feature_vector):
        """
        搜索叶子节点
        :param feature_vector: 特征向量
        :return: 预测值
        """
        current_node = self.tree.get_left_child()
        # 记录搜索路径
        search_path = []
        while current_node.get_leaf_node() is None:
            split_feature_id = current_node.get_split_feature_id()
            split_feature_value = current_node.get_split_feature_value()
            current_feature_value = feature_vector[split_feature_id]
            if current_feature_value <= split_feature_value:
                current_node = current_node.get_left_child()
            else:
                current_node = current_node.get_right_child()
        return current_node.get_leaf_node().get_predict_value()


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_excel("/Users/liuyang/Mars/project/data_set/ENB2012_data.xlsx", dtype=object)
    data = data.sample(frac=1)
    data = data.values
    X, Y = data[:, 0: -2], data[:, -2]
    # 训练模型
    model = Cart()
    model.fit(X, Y)
    # 预测
    prediction = model.predict(X)
    print(prediction)
    res = [Y[i] - prediction[i] for i in range(len(prediction))]
    from matplotlib import pyplot as plt
    plt.plot(Y)
    plt.plot(prediction)
    plt.show()


