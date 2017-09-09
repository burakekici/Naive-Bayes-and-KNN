"""
    Introduction to Artificial Intelligence
    Naive Bayes
    Due: 18.05.2017
"""
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


def main():
    data = load_iris()
    shuffled_data, shuffled_target = shuffle(data.data, data.target)
    digitized_shuffled_data = digitize_data(shuffled_data)

    for i in range(0, 5):  # naive bayes calculations
        Xtrain, Xtest, Ytrain, Ytest = split_dataset(digitized_shuffled_data, shuffled_target, i)
        feature_freq, class_freq = train_bayes_model(Xtrain, Ytrain)
        predicts = naive_bayes(Xtest, feature_freq, class_freq)
        score = metrics.accuracy_score(Ytest, predicts)
        print("Naive Bayes accuracy score for part " + str(i) + " is: {}".format(round(score * 100, 2)))


def digitize_data(data):
    result = []
    for i in range(0, len(data)):
        s = data[i]
        bins = np.linspace(s[3], s[0], num=10)
        result.append(np.digitize(s, bins))  # inds
    return result


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


def train_bayes_model(Xtrain, Ytrain):
    feature_freq = {}  # { (class,feature):frequency }
    class_freq = {}  # { class:frequency }

    for i in range(len(Ytrain)):
        class_freq[Ytrain[i]] = class_freq.get(Ytrain[i], 0) + 1
        for j in range(len(Xtrain[i])):
            feature_freq[(Ytrain[i], Xtrain[i][j])] = feature_freq.get((Ytrain[i], Xtrain[i][j]), 0) + 1

    return feature_freq, class_freq


def naive_bayes(Xtest, feature_freq, class_freq):
    predicts = []

    for i in range(len(Xtest)):  # calculate for each array in test set
        temp = Xtest[i]
        probabilities = {}

        # calculate probability for each class 0,1,2 and pick max
        # P(c|x) = P(x1|c).P(x2|c).P(x3|c).P(x4|c).P(c) then argmax
        for i in range(0, 3):
            V = 10
            prob = float(1) / float(3)  # P(c) is the probability of given class (one of 0,1,2)
            # calculate likelihood probabilities for each feature
            for feature in temp:
                # P(x1|c) = x1 count in C + 1 / feature count in C + V      {=Count(X1,C) + 1 / Count(C) + V}
                prob *= float(feature_freq.get((i, feature), 0) + 1) / float((class_freq.get(i, 0) * 4) + V)

            probabilities[i] = prob  # probabilities for 0,1,2

        best = max(probabilities, key=probabilities.get)  # pick max
        predicts.append(best)

    return predicts


if __name__ == "__main__":
    main()
