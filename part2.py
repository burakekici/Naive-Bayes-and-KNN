"""
    Introduction to Artificial Intelligence
    K-Nearest Neighbors
    Due: 18.05.2017
"""
from collections import Counter

import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


def main():
    data = load_iris()
    shuffled_data, shuffled_target = shuffle(data.data, data.target)

    for i in range(0, 5):  # knn calculations
        k = 1
        Xtrain, Xtest, Ytrain, Ytest = split_dataset(shuffled_data, shuffled_target, i)
        predicts = knn(Xtrain, Xtest, Ytrain, k)
        score = metrics.accuracy_score(Ytest, predicts)
        print("K-Nearest Neighbor accuracy score with k=" + str(k) + " for part " + str(i) + " is: {}".format(
            round(score * 100, 2)))


def split_dataset(data, target, option):
    fold = len(target) / 5  # 150/5 = 30
    Xtrain_list = []  # float values for train set (len:120)
    Xtest = []  # float values for test set (len:30)
    Ytrain_list = []  # targets for train set
    Ytest = []  # targets for test set

    for i in range(0, 5):
        if i == option:
            Xtest = data[fold * i:fold * (i + 1)]
            Ytest = target[fold * i:fold * (i + 1)]
        else:
            Xtrain_list.append(data[fold * i:fold * (i + 1)])
            Ytrain_list.append(target[fold * i:fold * (i + 1)])

    Xtrain = merge_Xset(Xtrain_list)  # convert train data set from 4*30 list to 1*120 list
    Ytrain = merge_Yset(Ytrain_list)  # convert train target set from 4*30 list to 1*120 list

    return Xtrain, Xtest, Ytrain, Ytest


def merge_Xset(Xtrain_list):
    Xtrain = Xtrain_list[0][0]
    for ls in Xtrain_list:
        for item in ls:
            Xtrain = np.vstack([Xtrain, item])
    Xtrain = np.delete(Xtrain, 0, 0)
    return Xtrain


def merge_Yset(Ytrain_list):
    Ytrain = Ytrain_list[0][0]
    for ls in Ytrain_list:
        for item in ls:
            Ytrain = np.hstack([Ytrain, item])
    Ytrain = np.delete(Ytrain, 0)
    return Ytrain


def knn(Xtrain, Xtest, Ytrain, k):
    predicts = []

    for item in Xtest:  # get nearest neighbor's class of each array in test set
        nearest = get_nearest(item, k, Xtrain, Ytrain)
        predicts.append(nearest)

    return predicts


def get_nearest(item, k, Xtrain, Ytrain):
    distances = []
    for i in range(len(Xtrain)):  # add distance as tuple (index,distance)
        distances.append((i, distance.euclidean(item, Xtrain[i])))

    distances = sorted(distances, key=lambda x: x[1])[0:k]  # get the first k elements from sorted list

    k_indices = []  # get only indices of the selected tuples above
    for i in range(k):
        k_indices.append(distances[i][0])

    k_classes = []  # get the corresponding classes
    for i in range(k):
        k_classes.append(Ytrain[k_indices[i]])

    ctr = Counter(k_classes)  # count class frequencies in k_classes list

    return ctr.most_common()[0][0]  # return the class with highest frequency


if __name__ == "__main__":
    main()
