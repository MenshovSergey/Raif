import pandas as pd
import sklearn.preprocessing as preproccesing
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import random

cat_coulmns = ["Event_type", "Insurer_type", "Owner_region", "Owner_type", "Sales_channel", "VEH_aim_use", "VEH_model",
               "VEH_type_name"]
drop_columns = ["bad", "claim_id"]
# names = ["RandomForest", "GBM", "AdaBoost", "SVM", "Dummy"]
# classifiers = [RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42),
#                AdaBoostClassifier(random_state=42, n_estimators=100),
#                svm.SVC(random_state=42), DummyClassifier(random_state=42)]

names = ["AdaBoost"]
classifiers = [AdaBoostClassifier(random_state=42, n_estimators=100)]

THRESHOLD_VEH_MODEL = 10

NEW_VEH_MODEL_NAME = "VEH_model_another"

prefix_for_remove = ["VEH_model", "Owner_region"]
thresholds = [30, 80]


def union_columns(data, prefix, new_name, threshold):
    small_columns = []
    another = [0] * len(data.values)
    for name_column in list(data):
        if prefix in name_column:
            values = list(data.get(name_column))
            count = Counter(values)[1]
            if count < threshold:
                small_columns.append(name_column)
                need_index = [i for i, x in enumerate(values) if x == 1]
                for i in need_index:
                    another[i] = 1
    data = data.drop(small_columns, 1)
    data[new_name] = another
    return data


def read_data():
    data_clean = pd.read_csv("data/all.csv", sep=';', decimal=",")
    data = pd.get_dummies(data_clean, columns=cat_coulmns)
    y = data.get("bad")
    data = data.drop(drop_columns, 1)
    data = data.fillna(method='pad')
    for prefix, threshold in zip(prefix_for_remove, thresholds):
        data = union_columns(data, prefix, prefix + "_another", threshold)
    return data, y, data_clean


def normalize_data(data):
    scaler = preproccesing.MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def show_metrics(y_test, y_pred):
    print("Accuracy : " + str(accuracy_score(y_test, y_pred)))
    print("...")
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("...")
    print("F1 measure : " + str(f1_score(y_test, y_pred, average=None)))


def get_FP(y_test, y_pred, y_test_indices):
    res = []
    for i, (y_t, y_p) in enumerate(zip(y_test, y_pred)):
        if y_p == 0 and y_t == 1:
            res.append(y_test_indices[i])
    return res


def compare_classifiers(X, Y, names, classifiers):
    indices = range(len(Y))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, indices, random_state=50,
                                                                                     test_size=0.3)

    # sklearn_pca = PCA(n_components=230)
    # X_train = sklearn_pca.fit_transform(X_train)
    # X_test = sklearn_pca.transform(X_test)

    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(name)
        show_metrics(y_test, y_pred)
        return get_FP(y_test, y_pred, indices_test)
        # print()


def feature_importance(X, Y, feature_names):
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))


def pca(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlim(0, 1000, 20)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()


def main():
    X, Y, data_clean = read_data()

    X_norm = normalize_data(X)
    X_norm_cut = []
    Y = list(Y)
    Y_cut = []
    count_0 = 0
    count_1 = 0
    random.seed(42)
    indices = []
    for i, v in enumerate(Y):
        if v == 0 and random.random() < 0.4:
            X_norm_cut.append(X_norm[i])
            Y_cut.append(0)
            count_0 += 1
            indices.append(i)
        elif v == 1:
            X_norm_cut.append(X_norm[i])
            Y_cut.append(1)
            count_1 += 1
            indices.append(i)
    Y = Y_cut
    X_norm = np.asarray(X_norm_cut)

    print("Count object 0 = " + str(count_0))
    print("Count object 1 = " + str(count_1))

    fp_indices = compare_classifiers(X_norm, Y, names, classifiers)
    fp = open("fp","w")
    target = open("target","w")
    target_values = list(data_clean.get("bad"))
    data_clean = data_clean.drop(drop_columns, 1)
    for i,v in enumerate(target_values):
        if v == 1:
            print(str(data_clean.values[i]).replace("\n", ""), file=target)
    for i in fp_indices:
        print(str(data_clean.values[indices[i]]).replace("\n",""), file=fp)
    fp.close()
    target.close()
    feature_importance(X_norm, Y, list(X))
    # pca(X_norm)


main()
