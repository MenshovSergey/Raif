from sklearn.model_selection import train_test_split, KFold, cross_val_score

from cascade import get_all_stat, show_res
from sklearn.ensemble import AdaBoostClassifier

from main import get_data_without_normalize

import numpy as np


def check_equals(a1, a2):
    for a in a1:
        for b in a2:
            q = a == b
            if q.min():
                print("all bad")

    print("end check equals")


def repeat_learn(A, Y):

    indices = range(len(Y))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(A, Y, indices, random_state=50,
                                                                                     test_size=0.3)


    for i in range(3):
        ada = AdaBoostClassifier(random_state=50, n_estimators=100)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        show_res("adaBoost", y_test, y_pred)
        tn, tp, fp, fn = get_all_stat(y_test, y_pred)

        new_y_train = []
        new_x_train = []
        new_x_test_0 = []
        new_y_test_0 = []
        new_x_test_1 = []
        new_y_test_1 = []
        for i, v in enumerate(y_train):
            if v == 1:
                if len(new_x_test_1) < len(fp):
                    new_x_test_1.append(X_train[i])
                    new_y_test_1.append(1)
                else:
                    new_x_train.append(X_train[i])
                    new_y_train.append(1)
            elif len(new_x_test_0) < len(fn):
                new_x_test_0.append(X_train[i])
                new_y_test_0.append(0)
            else:
                new_x_train.append(X_train[i])
                new_y_train.append(0)

        new_x_test_1.extend(new_x_test_0)
        new_y_test_1.extend(new_y_test_0)
        new_x_test = new_x_test_1
        new_y_test = new_y_test_1
        for i, v in enumerate(y_test):
            if v == 0 and i not in fn:
                new_x_test.append(X_test[i])
                new_y_test.append(0)
            elif v == 1 and i not in fp:
                new_x_test.append(X_test[i])
                new_y_test.append(1)
            else:
                new_x_train.append(X_test[i])
                new_y_train.append(v)

        check_equals(new_x_test, new_x_train)
        X_train = new_x_train
        y_train = new_y_train
        X_test = new_x_test
        y_test = new_y_test

    np.save("X_test", X_test)
    np.save("X_train", X_train)
    np.save("y_train", y_train)
    np.save("y_test", y_test)

    ada = AdaBoostClassifier(random_state=50, n_estimators=100)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    show_res("adaBoost", y_test, y_pred)
    return ada


def split_3_class(A, Y):

    indices = range(len(Y))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(A, Y, indices, random_state=50,
                                                                                     test_size=0.3)


    ada = AdaBoostClassifier(random_state=50, n_estimators=100)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    show_res("adaBoost", y_test, y_pred)
    tn, tp, fp, fn = get_all_stat(y_test, y_pred)

    X_train_list = X_train.tolist()
    y_train_list = y_train

    for i in range(int(0.7*len(fp))):
        X_train_list.append(X_test[fp[i]])
        y_train_list.append(2)

    for i in range(int(0.7*len(fn))):
        X_train_list.append(X_test[fn[i]])
        y_train_list.append(3)

    new_x_test = []
    new_y_test = []

    for i in range(int(0.7*len(fp)), len(fp)):
        new_x_test.append(X_test[fp[i]])
        new_y_test.append(2)

    for i in range(int(0.7*len(fn)), len(fn)):
        new_x_test.append(X_test[fn[i]])
        new_y_test.append(3)

    for i,v in enumerate(y_test):
        if i not in fp and i not in fn:
            new_x_test.append(X_test[i])
            new_y_test.append(v)

    X_train = X_train_list
    y_train = y_train_list
    X_test = new_x_test
    y_test = new_y_test


    ada = AdaBoostClassifier(random_state=50, n_estimators=100)
    ada.fit(np.asarray(X_train), y_train)
    y_pred = ada.predict(X_test)
    show_res("adaBoost", y_test, y_pred)
    return ada

def test():
    X_test = np.load("X_test.npy")
    X_train = np.load("X_train.npy")
    y_test = np.load("y_test.npy")
    y_train = np.load("y_train.npy")
    ada = AdaBoostClassifier(random_state=50, n_estimators=100)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    show_res("adaBoost", y_test, y_pred)


def cross_valid():
    # test()
    A, Y, _ = get_data_without_normalize()
    split_3_class(A,Y)
    # cut = 100
    # A_cut = A[-cut:]
    # Y_cut = Y[-cut:]
    # A = A[0:-cut]
    # Y = Y[0:-cut]
    # ada = repeat_learn(A, Y)
    # y_pred = ada.predict(A_cut)
    # show_res("adaBoost",Y_cut, y_pred)


cross_valid()
