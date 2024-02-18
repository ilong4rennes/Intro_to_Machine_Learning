import numpy as np
import sys
from math import log2

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.dataset = None

class ReadData():
    def __init__(self, infile):
        self.infile = infile
    
    def load_data(self):
        dataset = np.loadtxt(self.infile, dtype=int, delimiter='\t', skiprows=1)
        return dataset
    
    def get_features_names(self):
        dataset = np.loadtxt(self.infile, dtype=str, delimiter='\t', skiprows=0)
        features_names = dataset[0][:-1]
        return features_names

class DecisionTree:
    def __init__(self, infile, outfile, max_depth, print_out):
        self.root = Node()
        self.read_data = ReadData(infile)
        self.dataset = self.read_data.load_data()
        self.features_names = self.read_data.get_features_names()
        self.outfile = outfile
        self.print_out = print_out
        self.max_depth = max_depth
        self.predictions = None

    def extract_labels(self, dataset):
        labels = []
        for row in dataset:
            labels.append(row[-1])
        return labels
    
    def extract_features(self, dataset):
        num_features = len(dataset[0]) - 1
        all_features = []
        for feature_index in range(num_features):
            feature = []
            for row in dataset:
                feature.append(row[feature_index])
            all_features.append(feature)
        return all_features

    def entropy(self, feature):
        counts = np.bincount(feature)
        probabilities = counts / len(feature)
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * log2(p)
        # entropy = format(entropy, '.6f')
        return entropy
    
    def mutual_information(self, labels, feature):
        hy = self.entropy(labels)
        hy_x = 0
        for unique_value in np.unique(feature):
            count = 0
            values = []
            for index in range(len(feature)):
                if feature[index] == unique_value:
                    values.append(labels[index])
                    count += 1
            f_v = count / len(feature) # the fraction of data points where x_d=v
            hy_x += f_v * self.entropy(values)
        return hy - hy_x
        
    def best_feature(self, labels, features, used_features):
        max_mutual_info = 0
        best_feature_index = 0

        for feature_index in range(len(features)):
            if feature_index not in used_features:
                mutual_info = self.mutual_information(labels, features[feature_index])
                if mutual_info > max_mutual_info:
                    max_mutual_info = mutual_info
                    best_feature_index = feature_index
        return best_feature_index
    
    def split_dataset(self, dataset, feature_index):
        left_dataset, right_dataset = [], []
        for row in dataset:
            if row[feature_index] == 1:
                left_dataset.append(row)
            else:
                right_dataset.append(row)
        return left_dataset, right_dataset

    def majority_vote(self, labels):
        count1, count0 = 0, 0
        for row in labels:
            if row == 1:
                count1 += 1
            else:
                count0 += 1
        if count1 >= count0: return 1
        else:                return 0

    def all_labels_same(self, labels):
        return len(np.unique(labels)) == 1
    
    def all_features_splitted_on(self, features):
        return len(np.unique(features)) == 0

    def train(self):
        dataset = self.dataset
        used_features = set()
        self.root.dataset = dataset
        self.root = self.tree_recurse(dataset, self.root, 0, used_features)
        return self.root

    def tree_recurse(self, dataset, node, depth, used_features):
        labels = self.extract_labels(dataset)
        features = self.extract_features(dataset)

        # base case
        if (self.all_labels_same(labels)
            or self.all_features_splitted_on(features)
            or depth == int(self.max_depth)):
                node.vote = self.majority_vote(labels)
                return node
        
        # recursive case
        else:
            best_feature_index = self.best_feature(labels, features, used_features)
            
            if best_feature_index is None:
                node.vote = self.majority_vote(labels)
                return node

            node.attr = best_feature_index
            used_features.add(best_feature_index)

            left_dataset, right_dataset = self.split_dataset(dataset, best_feature_index)

            if len(left_dataset) > 0:
                node.left = Node()
                node.left.dataset = left_dataset
                self.tree_recurse(left_dataset, node.left, depth + 1, used_features.copy())

            if len(right_dataset) > 0:
                node.right = Node()
                node.right.dataset = right_dataset
                self.tree_recurse(right_dataset, node.right, depth + 1, used_features.copy())

            return node

    def predict_single(self, row, node):
        if node == None:
            return 1
        elif node.vote != None:
            return node.vote
        else:
            if node.attr == None:
                return 1
            elif row[node.attr] == 1:
                return self.predict_single(row, node.left)
            else:
                return self.predict_single(row, node.right)
    
    def predict(self, root):
        dataset = self.dataset
        predictions = []
        for row in dataset:
            pred_label = self.predict_single(row, root)
            predictions.append(pred_label)
        self.predictions = np.array(predictions)
        np.savetxt(self.outfile, predictions, fmt='%i', delimiter='\n')
        return predictions

    def error(self):
        error_count = 0
        for row_index in range(len(self.dataset)):
            if self.dataset[row_index][-1] != self.predictions[row_index]:
                error_count += 1
        error_rate = error_count / len(self.dataset)
        error_rate = format(error_rate, '.6f')
        return error_rate
    
    def label_count(self, node):
        # Assuming there is a method to extract labels from the node
        labels = self.extract_labels(node.dataset)
        count1, count0 = 0, 0
        for label in labels:
            if label == 1:
                count1 += 1
            else:
                count0 += 1
        return count1, count0
    
    def print_tree(self, parent_node, node, depth=0, feature_value=None):
        with open(self.print_out, 'w') as file:  # Open the file in append mode
            self._print_tree(file, parent_node, node, depth, feature_value)

    def _print_tree(self, file, parent_node, node, depth=0, feature_value=None):
        if node:
            features_names = self.features_names
            count1, count0 = self.label_count(node)

            indent = "| " * depth

            if feature_value is not None:  # for non-root nodes
                print(f"{indent}{features_names[parent_node.attr]} = {feature_value}: [{count0} 0/{count1} 1]", file=file)
            else:  # for the root node
                print(f"[{count0} 0/{count1} 1]", file=file)

            # Recursive calls for left and right subtrees
            if node.right:
                self._print_tree(file, node, node.right, depth + 1, 0)
            if node.left:
                self._print_tree(file, node, node.left, depth + 1, 1)  # Assuming a binary split, 1 for left
    
if __name__ == '__main__':
    train_infile = sys.argv[1] # tsv file
    test_infile = sys.argv[2] # tsv file
    max_depth = sys.argv[3] # int
    train_outfile = sys.argv[4] # txt file
    test_outfile = sys.argv[5] # txt file
    metrics_out = sys.argv[6] # txt file
    print_out = sys.argv[7] # txt file

    train = DecisionTree(train_infile, train_outfile, max_depth, print_out)
    train_root = train.train()
    train_predict = train.predict(train_root)
    train_error = train.error()
    train.print_tree(None, train_root)

    test = DecisionTree(test_infile, test_outfile, max_depth, print_out)
    test_root = test.train()
    test_predict = test.predict(train_root)
    test_error = test.error()

    with open(metrics_out, "w") as file:
        file.write(f"error(train): {train_error}")
        file.write("\n")
        file.write(f"error(test): {test_error}")

    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)