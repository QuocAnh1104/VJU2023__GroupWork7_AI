Kaggle Bot Account Detection using Random Forest

Overview
This project aims to automatically detect bot accounts on Kaggle by using a Random Forest classifier implemented from scratch in Python. Kaggle, a popular platform for data science competitions, is often subject to manipulation by bot accounts that distort the results of competitions, votes, and community interactions. The goal of this project is to create a machine learning model that can reliably identify bot accounts based on account metadata.

Key Features:
- Detects bot accounts on Kaggle using account activity features (followers, following, datasets, code, discussions, etc.).
- Random Forest model implemented from scratch using NumPy and Pandas.
- Performance evaluation using common classification metrics (accuracy, precision, recall, F1-score).

Motivation
Kaggle competitions, rankings, and community interactions are often affected by bot accounts that artificially inflate votes and rankings. These bots can undermine the fairness of competitions and community engagement. This project is designed to:
- Improve the fairness of voting and competition results on Kaggle by detecting bot accounts.
- Provide an educational example of implementing a Random Forest model from scratch, without relying on machine learning libraries like scikit-learn.

Dataset
The dataset used in this project is kaggle_bot_accounts_detection.csv

Model Overview
The model used in this project is a Random Forest classifier, an ensemble learning method that builds multiple decision trees and aggregates their predictions for more accurate results. The key steps include:
- Bootstrap Sampling: Each tree is trained on a random sample of the training data.
- Random Feature Selection: At each split in a tree, only a random subset of features is considered.
- Majority Voting: The final prediction is determined by the majority vote of all trees in the forest.

Decision Tree Implementation
The decision tree is built using the CART (Classification and Regression Trees) algorithm with Gini impurity as the splitting criterion. The algorithm recursively splits the data at each node to minimize Gini impurity, selecting the best feature and threshold for each split.

Random Forest
The Random Forest model consists of multiple decision trees, each trained with a different subset of data and features. This helps reduce overfitting and improves the generalization ability of the model.
