import numpy as np
import pandas as pd
from collections import Counter

def create_node(value=None):
    """Creates a tree node with a value."""
    return {'value': value, 'left': None, 'right': None, 'feature': None, 'threshold': None}

def is_leaf_node(node):
    """Checks if the node is a leaf node."""
    return node['value'] is not None

def determine_feature_types(X):
    """Determines the feature types of the dataset."""
    feature_types = []
    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i])
        if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
            feature_types.append("numerical")
        else:
            feature_types.append("categorical")
    return feature_types

def split_data(X, feature, threshold):
    """Splits the data into left and right subsets based on the threshold."""
    left_indices = np.where(X[:, feature] <= threshold)[0]
    right_indices = np.where(X[:, feature] > threshold)[0]
    return left_indices, right_indices

def calculate_entropy(y):
    """Calculates the entropy of a label array."""
    label_counts = Counter(y)
    total = len(y)
    entropy_value = 0.0
    for count in label_counts.values():
        p = count / total
        if p > 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

def calculate_gini_index(y):
    """Calculates the Gini index of a label array."""
    counts = Counter(y)
    total = len(y)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

def calculate_majority_error(y):
    """Calculates the majority error of a label array."""
    counts = Counter(y)
    total = len(y)
    return 1 - max(counts.values()) / total

def information_gain(y, left_y, right_y):
    """Calculates the information gain from a split."""
    n = len(y)
    parent_entropy = calculate_entropy(y)
    left_entropy = calculate_entropy(left_y)
    right_entropy = calculate_entropy(right_y)
    child_entropy = (len(left_y) / n) * left_entropy + (len(right_y) / n) * right_entropy
    return parent_entropy - child_entropy

def best_split(X, y, feature_types):
    """Finds the best feature and threshold to split on."""
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]

    for feature in range(n_features):
        if feature_types[feature] == "numerical":
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = split_data(X, feature, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        else:
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices, right_indices = split_data(X, feature, value)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=100):
    """Builds the decision tree using the ID3 algorithm."""
    n_samples, n_labels = len(y), len(np.unique(y))

    if depth >= max_depth or n_labels == 1:
        return create_node(value=Counter(y).most_common(1)[0][0])

    feature_types = determine_feature_types(X)
    best_feature, best_threshold = best_split(X, y, feature_types)

    if best_feature is None:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_indices, right_indices = split_data(X, best_feature, best_threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return create_node(value=Counter(y).most_common(1)[0][0])

    left_node = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_node = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)

    parent_node = create_node()
    parent_node['feature'] = best_feature
    parent_node['threshold'] = best_threshold
    parent_node['left'] = left_node
    parent_node['right'] = right_node

    return parent_node

def predict(instance, tree):
    """Predicts the label for a single instance."""
    if is_leaf_node(tree):
        return tree['value']
    if instance[tree['feature']] <= tree['threshold']:
        return predict(instance, tree['left'])
    else:
        return predict(instance, tree['right'])

def run_decision_tree(X_train, y_train, X_test, y_test, max_depth):
    """Runs the decision tree algorithm and prints the results."""
    metrics = {criterion: {'train': [], 'test': []} for criterion in ['information_gain', 'majority_error', 'gini']}

    for depth in range(1, max_depth + 1):
        for criterion in metrics.keys():
            tree = build_tree(X_train, y_train, max_depth=depth)
            y_train_pred = np.array([predict(x, tree) for x in X_train])
            y_test_pred = np.array([predict(x, tree) for x in X_test])

            train_accuracy = np.mean(y_train == y_train_pred)
            test_accuracy = np.mean(y_test == y_test_pred)

            metrics[criterion]['train'].append(round(1 - train_accuracy, 3))
            metrics[criterion]['test'].append(round(1 - test_accuracy, 3))

    # Output the results
    print("Depth | I.G(Train) | I.G(Test) | M.E(Train) | M.E(Test) | Gini(Train) | Gini(Test)")
    print("------+-------------+------------+-------------+------------+--------------+-----------")
    for depth in range(1, max_depth + 1):
        print(f"{depth:<5} | {metrics['information_gain']['train'][depth - 1]:<11} | "
              f"{metrics['information_gain']['test'][depth - 1]:<10} | "
              f"{metrics['majority_error']['train'][depth - 1]:<11} | "
              f"{metrics['majority_error']['test'][depth - 1]:<10} | "
              f"{metrics['gini']['train'][depth - 1]:<12} | "
              f"{metrics['gini']['test'][depth - 1]:<9}")

    for crit in ['information_gain', 'majority_error', 'gini']:
        min_error_depth = metrics[crit]['test'].index(min(metrics[crit]['test'])) + 1
        min_error_value = min(metrics[crit]['test'])
        print(f"Least {crit.replace('_', ' ').title()} error observed at depth {min_error_depth} with error {min_error_value:.3f}.")

# Load dataset
column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

df_train = pd.read_csv("Datasets/bank/train.csv", names=column_headers)
df_test = pd.read_csv("Datasets/bank/test.csv", names=column_headers)

X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

# Run the decision tree experiment
max_depth = 16
run_decision_tree(X_train, y_train, X_test, y_test, max_depth)
