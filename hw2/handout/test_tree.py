import numpy as np
from math import log2

class Node:
    def __init__(self):
        self.attribute = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.classification = None

class DecisionTree:
    def __init__(self, max_depth):
        self.root = Node()
        self.max_depth = max_depth

    def entropy(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum([p * log2(p) for p in probabilities if p > 0])
        return entropy

    def mutual_information(self, feature, labels):
        unique_values = np.unique(feature)
        feature_entropy = self.entropy(labels)
        
        conditional_entropy = 0.0
        for value in unique_values:
            subset_labels = labels[feature == value]
            conditional_entropy += (len(subset_labels) / len(labels)) * self.entropy(subset_labels)

        return feature_entropy - conditional_entropy

    def best_split(self, features, labels):
        best_mi = -1
        best_feature_index = None

        for i in range(features.shape[1]):
            mi = self.mutual_information(features[:, i], labels)
            if mi > best_mi:
                best_mi = mi
                best_feature_index = i

        return best_feature_index

    def build_tree(self, features, labels, depth=0):
        node = Node()
        
        if len(np.unique(labels)) == 1 or depth == self.max_depth:
            node.is_leaf = True
            node.classification = np.unique(labels)[0]
            return node

        best_feature_index = self.best_split(features, labels)
        if best_feature_index is None:
            node.is_leaf = True
            node.classification = np.bincount(labels).argmax()
            return node

        node.attribute = best_feature_index
        left_indices = features[:, best_feature_index] == 0
        right_indices = features[:, best_feature_index] == 1

        node.left = self.build_tree(features[left_indices], labels[left_indices], depth+1)
        node.right = self.build_tree(features[right_indices], labels[right_indices], depth+1)

        return node

    def fit(self, features, labels):
        self.root = self.build_tree(features, labels)

    def predict(self, features):
        predictions = [self._predict_single(feature, self.root) for feature in features]
        return np.array(predictions)

    def _predict_single(self, feature, node):
        if node.is_leaf:
            return node.classification

        if feature[node.attribute] == 0:
            return self._predict_single(feature, node.left)
        else:
            return self._predict_single(feature, node.right)

# Example usage
# dataset = load_your_dataset() # Replace with your dataset loading method
# features, labels = dataset[:, :-1], dataset[:, -1]
# tree = DecisionTree(max_depth=5)
# tree.fit(features, labels)
# predictions = tree.predict(features)
