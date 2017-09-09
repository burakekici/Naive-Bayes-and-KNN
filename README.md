# Naive-Bayes-and-KNN

In this project, Naive Bayes classifier and K-Nearest Neighbors classifier are implemented with using Iris dataset.

# First steps (applied for both algorithms)

1. Numpy, scipy and sklearn libraries are downloaded.
2. Dataset is shuffled.
3. 5-fold cross validation is implemented. (4 parts are train set; 1 part is test set)

# part1.py - Naive Bayes Classifier

4. Iris dataset values are converted from float to integer.
5. The model is trained.
6. The class is predicted for each item in test set.

# part2.py - K-Nearest Neighbors Classifier

4. K-Nearest Neighbors are calculated.

To calculate KNN for k=3, it is needed to calculate the nearest three neighbors of the given item. Euclidian distance function is used to do this. Then, the most frequent class value is selected for the item in test set.
