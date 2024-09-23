import pandas as pd
import numpy as np
import math

train_data_car = pd.read_csv('Datasets/car/train.csv', header=None)
test_data_car = pd.read_csv('Datasets/car/test.csv', header=None)

column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train_data_car.columns = column_names
test_data_car.columns = column_names

def entropy_car(y):
    values_car, counts_car = np.unique(y, return_counts=True)
    probabilities_car = counts_car / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities_car])

def information_gain_car(X, y, feature):
    total_entropy_car = entropy_car(y)
    values_car, counts_car = np.unique(X[feature], return_counts=True)

    weighted_entropy = np.sum([(counts_car[i] / np.sum(counts_car)) * entropy_car(y[X[feature] == values_car[i]]) for i in range(len(values_car))])

    return total_entropy_car - weighted_entropy

def gini_index_car(y):
    values_car, counts_car = np.unique(y, return_counts=True)
    probabilities_car = counts_car / len(y)
    return 1 - np.sum([p ** 2 for p in probabilities_car])

def gini_gain_car(X, y, feature):
    total_gini_car = gini_index_car(y)
    values_car, counts_car = np.unique(X[feature], return_counts=True)

    weighted_gini = np.sum([(counts_car[i] / np.sum(counts_car)) * gini_index_car(y[X[feature] == values_car[i]]) for i in range(len(values_car))])

    return total_gini_car - weighted_gini

def majority_error_car(y):
    values_car, counts_car = np.unique(y, return_counts=True)
    majority_class_count_car = np.max(counts_car)
    return 1 - (majority_class_count_car / len(y))

def majority_error_gain_car(X, y, feature):
    total_error_car = majority_error_car(y)
    values_car, counts_car = np.unique(X[feature], return_counts=True)

    weighted_error_car = np.sum([(counts_car[i] / np.sum(counts_car)) * majority_error_car(y[X[feature] == values_car[i]]) for i in range(len(values_car))])

    return total_error_car - weighted_error_car

class DecisionTreeID3_car:
    def __init__(self, max_depth=None, parameter='information_gain'):
        self.max_depth = max_depth
        self.parameter = parameter
        self.tree = None

    def fit(self, X, y):
        self.tree = self._id3(X, y, depth=0)

    def _id3(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]

        if self.max_depth is not None and depth >= self.max_depth or X.empty:
            return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]

        best_feature_car = self._choose_best_feature(X, y)

        tree = {best_feature_car: {}}

        for value in np.unique(X[best_feature_car]):
            X_subset = X[X[best_feature_car] == value].drop(columns=[best_feature_car])
            y_subset = y[X[best_feature_car] == value]
            subtree = self._id3(X_subset, y_subset, depth + 1)
            tree[best_feature_car][value] = subtree

        return tree

    def _choose_best_feature(self, X, y):
        if self.parameter == 'information_gain':
            gains = [information_gain_car(X, y, feature) for feature in X.columns]
        elif self.parameter == 'gini_index':
            gains = [gini_gain_car(X, y, feature) for feature in X.columns]
        elif self.parameter == 'majority_error':
            gains = [majority_error_gain_car(X, y, feature) for feature in X.columns]

        return X.columns[np.argmax(gains)]

    def predict_car(self, X):
        return X.apply(self._predict_row_car, axis=1)

    def _predict_row_car(self, row):
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))
            value = row[feature]
            if value in node[feature]:
                node = node[feature][value]
            else:
                return None
        return node

X_train_car = train_data_car.iloc[:, :-1]
y_train_car = train_data_car.iloc[:, -1]


X_test_car = test_data_car.iloc[:, :-1]
y_test_car = test_data_car.iloc[:, -1]

tree = DecisionTreeID3_car(max_depth=5, parameter='information_gain')
tree.fit(X_train_car, y_train_car)

predictions = tree.predict_car(X_test_car)

accuracy = np.sum(predictions == y_test_car) / len(y_test_car)
print(f'Accuracy: {accuracy * 100:.2f}%')


def prediction_error_car(y_true, y_pred):
    return np.mean(y_true != y_pred)

depth_range = range(1, 7)
parameter = ['information_gain', 'gini_index', 'majority_error']

results2 = []

for depth in depth_range:
    for param in parameter:
        tree = DecisionTreeID3_car(max_depth=depth, parameter=param)

        tree.fit(X_train_car, y_train_car)

        train_predictions = tree.predict_car(X_train_car)
        test_predictions = tree.predict_car(X_test_car)

        train_error = prediction_error_car(y_train_car, train_predictions)
        test_error = prediction_error_car(y_test_car, test_predictions)

        results2.append({
            'depth': depth,
            'criterion': param,
            'train_error': train_error,
            'test_error': test_error
        })

results_df = pd.DataFrame(results2)

print(results_df)