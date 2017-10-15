# import main.py
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from main import read_data, normalize_data, THRESHOLD_DATA, feature_importance, compare_classifiers, names, classifiers, \
    union_data, union_columns2, union_columns

prefix_for_remove2 = ["Owner_region","VEH_model"]

factors = [['Республика Дагестан', 'Ростовская область', 'Республика Татарстан (Татарстан)',
            "Москва", 'Волгоградская область', 'Челябинская область', "Краснодарский край", "Московская область",
            "Новосибирская область", 'Республика Башкортостан'],
           ["MERCEDES","BMW","AUDI","PORSCHE","LAND ROVER","LEXUS","INFINITI","CADILLAC","TOYOTA LAND CRUISER"]]

effect = []

res = 0
best_city = []

def rec(k, j, X, Y):
    if j >= len(factors[k]):
        new = []
        for i in range(len(factors[k])):
            if effect[i] > 0:
                new.append(factors[k][i])
        A = union_columns2(X, prefix_for_remove2[k], prefix_for_remove2[k] + "_another", new)
        A = feature_importance(A.values, Y, list(A))
        accuracy = compare_classifiers(A, Y, names, classifiers)
        global res
        if accuracy > res:
            res = accuracy
            print(res)
            global best_city
            best_city = new
    else:
        for i in range(0, 2):
            effect[j] = i
            rec(k, j + 1, X, Y)

def show_metrics(y_test, y_pred):
    print("...")
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    arr = f1_score(y_test, y_pred, average=None)
    return arr[1]


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
        return show_metrics(y_test, y_pred)

def main():
    X, Y, data_clean = read_data()
    X = union_data(X)
    Y = list(Y)
    Y_cut = []
    random.seed(42)
    indices = []
    remove_indices= []
    for i, v in enumerate(Y):
        if v == 0 and random.random() < THRESHOLD_DATA or v == 1:
            indices.append(i)
            Y_cut.append(v)
        else:
            remove_indices.append(i)
    Y = Y_cut
    X_norm = X.drop(X.index[remove_indices])

    #print("Count object 0 = " + str(count_0))
    #print("Count object 1 = " + str(count_1))

    # A = feature_importance(X_norm, Y, list(X))
    #print("A:", len(A[0]))
    global effect
    effect = [0] * len(factors[0])
    rec(0, 0, X_norm, Y)
    print(res)
    print(best_city)

main()