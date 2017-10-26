import random

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split

from main import get_data_without_normalize, read_data, union_data, THRESHOLD_DATA, union_columns2, union_columns3, \
    prefix_for_remove2, factors, COUNT

import numpy as np

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
    res_names = []
    for i in range(COUNT+1):
        res_names.append(feature_names[indices[i]])

    A = new_data(indices, importances, X)

    return A, res_names


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
    A, res_names = feature_importance(A.values, Y, list(A))
    return A, Y, res_names

def draw():
    experts = [15,25,55,75,80,82,90]
    AI =      [55,65,72,73,75,77,80]
    X =       [10,20,30,40,50,60,70]
    plt.plot(X, experts, color='blue', label='Эксперты')
    plt.plot(X,AI, color='green', label='AI')
    plt.xlabel("Финансовые затраты")
    plt.ylabel("Точность")

    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.title("Зависимость точности от финансовых затрат")
    plt.show()

# draw()

def tree_interpreter():
    A_clean, Y,f_names = read_and_get()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(A_clean, Y,  random_state=50, test_size=0.3)
    rf.fit(X_train, y_train)
    prediction, bias, contributions = ti.predict(rf, X_test)

    for i in range(10):
        print ("Instance", i)
        print ("Bias (trainset mean)", bias[i])
        print ("Feature contributions:")

        for c, feature in (zip(contributions[i],f_names)):
            print (feature, round(c[np.argmax(prediction[i])], 2))


tree_interpreter()
