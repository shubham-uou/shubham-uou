import numpy as np
import pandas as pd
from collections import Counter

def create_leaf(value):
    """Creates a leaf node with the given value."""
    return {'value': value, 'left': None, 'right': None, 'feature': None, 'threshold': None}

def is_leaf(node):
    """Checks if a node is a leaf."""
    return node['value'] is not None

def identify_feature_types(X):
    """Determines if features are numerical or categorical."""
    feature_types = []
    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i])
        if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
            feature_types.append("numerical")
        else:
            feature_types.append("categorical")
    return feature_types

def impute_missing_values(X):
    """Imputes missing (unknown) values with the most common values."""
    majority_values = {}
    for i in range(X.shape[1]):
        if 'unknown' in X[:, i]:
            majority_value = Counter(X[X[:, i] != 'unknown', i]).most_common(1)[0][0]
            majority_values[i] = majority_value
    return majority_values

def replace_missing_values(X, majority_values):
    """Replaces missing values in X with the imputed values."""
    X_imputed = X.copy()
    for i in majority_values:
        X_imputed[X[:, i] == 'unknown', i] = majority_values[i]
    return X_imputed

def split_dataset(X, feature, threshold):
    """Splits the dataset based on a feature and threshold."""
    left_indices = np.where(X[:, feature] <= threshold)[0]
    right_indices = np.where(X[:, feature] > threshold)[0]
    return left_indices, right_indices

def calculate_entropy(y):
    """Calculates the entropy of a label array."""
    label_counts = Counter(y)
    total = len(y)
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * np.log2(probability)
    return entropy

def calculate_gini_index(y):
    """Calculates the Gini index for a label array."""
    counts = Counter(y)
    total = len(y)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

def calculate_majority_error(y):
    """Calculates the majority error for a label array."""
    counts = Counter(y)
    total = len(y)
    return 1 - max(counts.values()) / total

def information_gain(y, left_y, right_y):
    """Calculates the information gain from a split."""
    n = len(y)
    if n == 0:
        return 0
    parent_entropy = calculate_entropy(y)
    left_entropy = calculate_entropy(left_y)
    right_entropy = calculate_entropy(right_y)
    child_entropy = (len(left_y) / n) * left_entropy + (len(right_y) / n) * right_entropy
    return parent_entropy - child_entropy

def find_best_split(X, y, feature_types):
    """Finds the best feature and threshold to split on."""
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    num_features = X.shape[1]

    for feature in range(num_features):
        if feature_types[feature] == "numerical":
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = split_dataset(X, feature, threshold)
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
                if value == 'unknown':
                    continue
                left_indices, right_indices = split_dataset(X, feature, value)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=100):
    """Recursively builds the decision tree."""
    n_samples, n_labels = len(y), len(np.unique(y))

    if depth >= max_depth or n_labels == 1:
        return create_leaf(Counter(y).most_common(1)[0][0])

    feature_types = identify_feature_types(X)
    best_feature, best_threshold = find_best_split(X, y, feature_types)

    if best_feature is None:
        return create_leaf(Counter(y).most_common(1)[0][0])

    left_indices, right_indices = split_dataset(X, best_feature, best_threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return create_leaf(Counter(y).most_common(1)[0][0])

    left_subtree = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_subtree = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)

    node = create_leaf(None)  # Create a node that will hold the split info
    node['feature'] = best_feature
    node['threshold'] = best_threshold
    node['left'] = left_subtree
    node['right'] = right_subtree

    return node

def traverse_tree(instance, tree):
    """Traverses the tree to predict the label for a single instance."""
    if is_leaf(tree):
        return tree['value']
    if instance[tree['feature']] <= tree['threshold']:
        return traverse_tree(instance, tree['left'])
    else:
        return traverse_tree(instance, tree['right'])

def run_decision_tree(X_train, y_train, X_test, y_test, max_depth):
    """Runs the decision tree and evaluates its performance."""
    metrics = {criterion: {'train': [], 'test': []} for criterion in ['information_gain', 'majority_error', 'gini']}
    best_depths = {criterion: 0 for criterion in metrics.keys()}
    best_accuracies = {criterion: 0 for criterion in metrics.keys()}

    majority_values = impute_missing_values(X_train)
    X_train = replace_missing_values(X_train, majority_values)
    X_test = replace_missing_values(X_test, majority_values)

    for depth in range(1, max_depth + 1):
        for criterion in metrics.keys():
            tree = build_tree(X_train, y_train, max_depth=depth)
            y_train_pred = np.array([traverse_tree(x, tree) for x in X_train])
            y_test_pred = np.array([traverse_tree(x, tree) for x in X_test])

            train_accuracy = np.mean(y_train == y_train_pred)
            test_accuracy = np.mean(y_test == y_test_pred)

            metrics[criterion]['train'].append(round(1 - train_accuracy, 3))
            metrics[criterion]['test'].append(round(1 - test_accuracy, 3))

            if test_accuracy > best_accuracies[criterion]:
                best_accuracies[criterion] = test_accuracy
                best_depths[criterion] = depth

    # Print results
    print("\nResults Summary:")
    print(f"{'Depth':<6} {'I.G(Train)':<12} {'I.G(Test)':<12} {'M.E(Train)':<12} {'M.E(Test)':<12} {'Gini(Train)':<12} {'Gini(Test)':<12}")
    for depth in range(1, max_depth + 1):
        print(f"{depth:<6} {metrics['information_gain']['train'][depth - 1]:<12} {metrics['information_gain']['test'][depth - 1]:<12} "
              f"{metrics['majority_error']['train'][depth - 1]:<12} {metrics['majority_error']['test'][depth - 1]:<12} "
              f"{metrics['gini']['train'][depth - 1]:<12} {metrics['gini']['test'][depth - 1]:<12}")

    for crit in ['information_gain', 'majority_error', 'gini']:
        print(f"Least {crit.replace('_', ' ').title()} error observed at depth {best_depths[crit]} with error {1 - best_accuracies[crit]:.3f}.")

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

