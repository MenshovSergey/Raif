import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as preproccesing
from sklearn import neighbors, svm
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

cat_coulmns = ["Event_type", "Insurer_type", "Owner_region", "Owner_type", "Sales_channel", "VEH_aim_use", "VEH_model",
               "VEH_type_name"]
drop_columns = ["bad", "claim_id"]
names = ["RandomForest", "GBM", "AdaBoost", "SVM", "Dummy"]
classifiers = [RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42),
               AdaBoostClassifier(random_state=42, n_estimators=100),
               svm.SVC(random_state=42), DummyClassifier(random_state=42)]

# names = ["AdaBoost"]
# classifiers = [AdaBoostClassifier(random_state=42, n_estimators=100)]
# classifiers = [neighbors.KNeighborsClassifier(weights="distance", algorithm='kd_tree',n_neighbors=10)]

THRESHOLD_VEH_MODEL = 10
THRESHOLD_FI = 0.01
THRESHOLD_DATA = 0.1
COUNT = 25

NEW_VEH_MODEL_NAME = "VEH_model_another"

prefix_for_remove = ["VEH_model"]
# prefix_for_remove2 = ["Owner_region"]
prefix_for_remove2 = ["Owner_region", "VEH_model"]
# factors = [['Ростовская область', 'Москва', 'Волгоградская область', 'Челябинская область', 'Краснодарский край']]
factors = [['Ростовская область', 'Москва', 'Волгоградская область', 'Челябинская область', 'Краснодарский край'],
           ['MERCEDES', 'BMW', 'AUDI']]
thresholds = [30]


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


def union_columns2(data, prefix, new_name, main_names):
    small_columns = []
    another = [0] * len(data.values)
    for name_column in list(data):
        if prefix in name_column:
            flag = False
            for name in main_names:
                if name in name_column:
                    flag = True
                    break
            if not flag:
                values = list(data.get(name_column))
                small_columns.append(name_column)
                need_index = [i for i, x in enumerate(values) if x == 1]
                for i in need_index:
                    another[i] = 1

    res = data.drop(small_columns, 1)
    res[new_name] = another
    return res


def union_data(data):
    for prefix, threshold in zip(prefix_for_remove, thresholds):
        data = union_columns(data, prefix, prefix + "_another", threshold)

    for prefix, factor_data in zip(prefix_for_remove2, factors):
        data = union_columns2(data, prefix, prefix + "_another", factor_data)
    return data


def union_columns3(data, prefix, new_name, main_names):
    another0 = [0] * len(data.values)
    another1 = [0] * len(data.values)
    drop_pos = []
    for name_column in list(data):
        if prefix in name_column:
            drop_pos.append(name_column)
            flag = 0
            for name in main_names:
                if name in name_column:
                    flag = 1
                    break
            val = list(data.get(name_column))
            need = [i for i, x in enumerate(val) if x == 1]
            if flag == 1:
                for i in need:
                    another1[i] = 1
            else:
                for i in need:
                    another0[i] = 1
    # print(len(another0), " ", sum(another0))
    # print(len(another1), " ", sum(another1))
    # print("...")
    res = data.drop(drop_pos, 1)
    new_name0 = new_name + "0"
    res[new_name0] = another0
    new_name1 = new_name + "1"
    res[new_name1] = another1
    return res


def read_data():
    data_clean = pd.read_csv("data/all.csv", sep=';', decimal=",")
    data = pd.get_dummies(data_clean, columns=cat_coulmns)
    y = data.get("bad")
    data = data.drop(drop_columns, 1)
    data = data.fillna(method='pad')
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


def save(model):
    f = open("ada", "wb")
    import pickle
    pickle.dump(model, f)
    f.close()


def compare_classifiers(X, Y, names, classifiers):
    indices = range(len(Y))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, indices, random_state=50,
                                                                                     test_size=0.3)

    # sklearn_pca = PCA(n_components=230)
    # X_train = sklearn_pca.fit_transform(X_train)
    # X_test = sklearn_pca.transform(X_test)

    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, y_train)
        save(classifier)
        y_pred = classifier.predict(X_test)
        print(name)
        show_metrics(y_test, y_pred)
        get_FP(y_test, y_pred, indices_test)
        # print()


def new_data(pos, importance, X):
    i = len(pos) - 1
    # while importance[pos[i]] < THRESHOLD_FI:
    #     pos = np.delete(pos, i, 0)
    #     i -= 1
    while i > COUNT:
        pos = np.delete(pos, i, 0)
        i -= 1

    A = np.ndarray((len(X), len(pos)))
    for i in range(len(pos)):
        A[:, i] = X[:, pos[i]]
    return A


def feature_importance(X, Y, feature_names):
    adabBoost = AdaBoostClassifier(random_state=42, n_estimators=100)
    adabBoost.fit(X, Y)
    importances = adabBoost.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    A = new_data(indices, importances, X)
    return A


def pca(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlim(0, 200, 20)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()


def get_stat(data):
    res = {}
    for v in data:
        for i, p in enumerate(v):
            if not i in res:
                res[i] = Counter()
            res[i].update([p])
    return res


def print_stat(f, stat):
    for k, v in stat.items():
        print(v, file=f)


def read_and_get():
    X, Y, data_clean = read_data()
    X = union_data(X)
    Y = list(Y)
    Y_cut = []
    count_0 = 0
    count_1 = 0
    random.seed(42)
    indices = []
    remove_indices = []
    for i, v in enumerate(Y):
        if v == 0 and random.random() < THRESHOLD_DATA:
            # X_norm_cut.append(X_norm[i])
            Y_cut.append(0)
            count_0 += 1
            indices.append(i)
        elif v == 1:
            # X_norm_cut.append(X_norm[i])
            Y_cut.append(1)
            count_1 += 1
            indices.append(i)
        else:
            remove_indices.append(i)
    Y = Y_cut
    X_norm = X.drop(X.index[remove_indices])
    X_norm.to_csv("union_filter.csv", sep=",")

    print("Count object 0 = " + str(count_0))
    print("Count object 1 = " + str(count_1))

    A = union_columns2(X_norm, prefix_for_remove2[0], prefix_for_remove2[0] + "_another", factors[0])
    A = union_columns3(A, prefix_for_remove2[1], prefix_for_remove2[1] + "_another_car", factors[1])
    # A = union_agents(A)
    A = feature_importance(A.values, Y, list(A))
    return A, Y


def get_data():
    A, Y = read_and_get()
    A = normalize_data(A)
    return A, Y, range(len(Y))


def get_data_without_normalize():
    A, Y = read_and_get()
    return A, Y, range(len(Y))


def main():
    A, Y, _ = get_data()
    fp_indices = compare_classifiers(A, Y, names, classifiers)
    # fp_indices2 = compare_classifiers(A, Y, names, classifiers)

# main()
