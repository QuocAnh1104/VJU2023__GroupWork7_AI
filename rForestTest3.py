import numpy as np
import pandas as pd
import math
from typing import Optional


# ================================
# 1. Decision Tree implementation
# ================================

class DecisionTreeNode:
    """
    A single node in the decision tree.
    """
    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self,
                 feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: 'DecisionTreeNode' = None,
                 right: 'DecisionTreeNode' = None,
                 value: Optional[int] = None):
        self.feature_index = feature_index  # index of feature to split on
        self.threshold = threshold          # numeric split threshold
        self.left = left                    # left child (<= threshold)
        self.right = right                  # right child (> threshold)
        self.value = value                  # class label at leaf (0/1)


class DecisionTreeClassifierScratch:
    """
    Simple CART-style decision tree for classification using Gini impurity.
    Used as base learner for Random Forest.
    """
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # number of features to consider at each split
        self.n_classes_ = None
        self.n_features_ = None
        self.root = None

    # ---------- impurity ----------
    def _gini(self, y):
        m = len(y)  # y is a 1D array (or list) of class labels at the current node
        _, counts = np.unique(y, return_counts=True) # Finds all unique class labels in y, also returns how many times each label appears
        probs = counts / m # counts = array of class counts.
        return 1.0 - np.sum(probs ** 2)

    # ---------- majority class ----------
    def _majority_class(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    # ---------- best split search ----------
    def _best_split(self, X, y):
        m, n = X.shape  # m is the number of rows (or samples) and n is number of column (or features)
        if m < self.min_samples_split:
            return None, None, None, None, None

        parent_impurity = self._gini(y)
        best_gain = 0.0
        best_idx = None
        best_thr = None
        best_left_mask = None
        best_right_mask = None

        # choose subset of features at this node (for Random Forest behaviour)
        feature_indices = np.arange(n) #create an array of feature index 
        if self.max_features is not None and self.max_features < n:
            feature_indices = np.random.choice(feature_indices, self.max_features, replace=False)

        for idx in feature_indices:
            X_col = X[:, idx]

            # sort values for efficient threshold generation
            sorted_idx = np.argsort(X_col)  # give the sorted index that make the values in ascending order (5,2,8: value) -> (1,0,2: sorted value)
            X_sorted = X_col[sorted_idx]
            y_sorted = y[sorted_idx]

            # candidate thresholds: midpoints where value AND class change
            for i in range(1, m):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                if y_sorted[i] == y_sorted[i - 1]:
                    continue

                thr = (X_sorted[i] + X_sorted[i - 1]) / 2.0
                left_mask = X_col <= thr
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                g_left = self._gini(y_left)
                g_right = self._gini(y_right)

                n_left = len(y_left) # (the number of sameples in left node)
                n_right = len(y_right) # the number of samples in right notde  
                impurity = (n_left / m) * g_left + (n_right / m) * g_right
                gain = parent_impurity - impurity

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = thr
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        return best_idx, best_thr, best_gain, best_left_mask, best_right_mask

    # ---------- recursive tree building ----------
    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_labels = len(np.unique(y))

        # stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) \
           or num_labels == 1 \
           or num_samples < self.min_samples_split:
            leaf_value = self._majority_class(y)
            return DecisionTreeNode(value=leaf_value)

        idx, thr, gain, left_mask, right_mask = self._best_split(X, y)
        if idx is None or gain is None or gain <= 0:
            leaf_value = self._majority_class(y)
            return DecisionTreeNode(value=leaf_value)

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)   # the final output is the leafnode with the determined label
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_index=idx,
            threshold=thr,
            left=left_child,
            right=right_child
        )

    # ---------- public API ----------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, row):
        node = self.root
        while node.value is None:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(row) for row in X])


# =================================
# 2. Random Forest implementation
# =================================

class RandomForestClassifierScratch:
    """
    Random Forest classifier from scratch using the DecisionTreeClassifierScratch above.
    """
    def __init__(self, n_trees=3, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features   # "sqrt", "log2", or int
        self.trees = []
        self.n_features_ = None

    def _get_n_sub_features(self, n_features):
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(math.log2(n_features)))
        return n_features

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.trees = []

        n_sub_features = self._get_n_sub_features(n_features)  # the number of features chosen for each tree 

        for _ in range(self.n_trees):
            # bootstrap sample
            indices = np.random.randint(0, n_samples, size=n_samples)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=n_sub_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)

        n_samples = X.shape[0]
        final_preds = np.empty(n_samples, dtype=all_preds.dtype)

        for i in range(n_samples):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds[i] = values[np.argmax(counts)]
        return final_preds


# ======================================
# 3. Helper: metrics from confusion matrix
# ======================================

def confusion_matrix_binary(y_true, y_pred):
    """
    Returns TP, FP, FN, TN for binary labels (0/1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, fp, fn, tn


def classification_report_binary(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ======================================
# 4. Load data, preprocess, train & evaluate
# ======================================

def main():
    # ---- 4.1 Load dataset ----
    # Make sure the CSV is in the same folder or adjust the path
    df = pd.read_csv("kaggle_bot_accounts_detection.csv")

    # Only use the specified attributes + target
    FEATURES = [
        "IS_GLOGIN",
        "FOLLOWER_COUNT",
        "FOLLOWING_COUNT",
        "DATASET_COUNT",
        "CODE_COUNT",
        "DISCUSSION_COUNT",
        "AVG_NB_READ_TIME_MIN",
    ]
    TARGET = "ISBOT"

    data = df[FEATURES + [TARGET]].copy()   # extract the table with only desired features

    # ---- 4.2 Basic preprocessing ----
    # Handle IS_GLOGIN (boolean + NaN) -> 0/1
    if data["IS_GLOGIN"].dtype != bool:
        # if it's object or float with True/False/NaN, cast via bool
        mode = data["IS_GLOGIN"].mode(dropna=True) # dropna=True means ignore NaN when computing the mode.
        fill_val = bool(mode.iloc[0]) if not mode.empty else False
        data["IS_GLOGIN"] = data["IS_GLOGIN"].fillna(fill_val).astype(bool)
    data["IS_GLOGIN"] = data["IS_GLOGIN"].astype(int)  # True->1, False->0

    # Numeric columns: fill NaN with median
    num_cols = [
        "FOLLOWER_COUNT",
        "FOLLOWING_COUNT",
        "DATASET_COUNT",
        "CODE_COUNT",
        "DISCUSSION_COUNT",
        "AVG_NB_READ_TIME_MIN",
    ]
    for col in num_cols:
        med = data[col].median()
        data[col] = data[col].fillna(med)

    # Target: boolean -> 0/1
    data[TARGET] = data[TARGET].astype(int)

     # ---- 4.3 Stratified train-test split (holdout) ----
    X = data[FEATURES].values
    y = data[TARGET].values

    n_samples = len(y)

    # indices of each class
    idx_bot = np.where(y == 1)[0]
    idx_nonbot = np.where(y == 0)[0]

    # shuffle each group separately
    np.random.shuffle(idx_bot)
    np.random.shuffle(idx_nonbot)

    # 80/20 split inside each class
    split_bot = int(0.8 * len(idx_bot))
    split_nonbot = int(0.8 * len(idx_nonbot))

    train_idx = np.concatenate([idx_bot[:split_bot], idx_nonbot[:split_nonbot]])
    test_idx  = np.concatenate([idx_bot[split_bot:], idx_nonbot[split_nonbot:]])

    # shuffle final train & test indices (optional, for randomness)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # (optional) check class ratios
    def ratio_info(name, labels):
        total = len(labels)
        bots = labels.sum()
        nonbots = total - bots
        print(
            f"{name}: total={total}, "
            f"bots={bots} ({bots/total:.3%}), "
            f"non-bots={nonbots} ({nonbots/total:.3%})"
        )

    ratio_info("All data ", y)
    ratio_info("Train set", y_train)
    ratio_info("Test set ", y_test)


    # ---- 4.4 Train Random Forest (from scratch) ----
    rf = RandomForestClassifierScratch(
        n_trees=10,
        max_depth= None,
        min_samples_split=2,
        max_features="sqrt",   # random subset of features at each split
    )
    rf.fit(X_train, y_train)

    # ---- 4.5 Evaluate ----
    y_pred = rf.predict(X_test)
    report = classification_report_binary(y_test, y_pred)

    print("=== Confusion Matrix ===")
    print(f"TP: {report['TP']}")
    print(f"FP: {report['FP']}")
    print(f"FN: {report['FN']}")
    print(f"TN: {report['TN']}")

    print("\n=== Metrics ===")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1-score : {report['f1']:.4f}")


if __name__ == "__main__":
    main()
