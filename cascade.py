from keras.models import model_from_json
import pickle

from sklearn import manifold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score
from NN import get_data
from matplotlib import pyplot as plt
import numpy as np

from main import get_data_without_normalize

SPLIT = 14
def load_nn(name, w):
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(w)
    return loaded_model

def load_classifier(c2):
    return pickle.load(open(c2,"rb"))

def compare_proba(y_pred, y_proba, y_test):
    res = []
    for i, v in enumerate(y_pred):
        if y_test[i] != v:
            print("true answer = "+str(y_test[i]))
            print(y_proba[i])


def show_res(name, y_test, y_res):
    print("\n")
    print(name+"\n")
    print(confusion_matrix(y_test, y_res))
    print("\n")
    print("F1 measure : " + str(f1_score(y_test, y_res, average=None)))

def get_all_stat(y_test, y_pred):
    #tn fn
    #fp tp
    tn = []
    tp = []
    fp = []
    fn = []
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i] == 0:
            tn.append(i)
        if y_test[i] == y_pred[i] == 1:
            tp.append(i)
        if y_test[i] == 0 and y_pred[i] == 1:
            fn.append(i)
        if y_test[i] == 1 and y_pred[i] == 0:
            fp.append(i)

    return tn, tp, fp, fn

def get_need_points(ind, pos):
    res = []
    for i in ind:
        res.append(pos[i])
    return np.asarray(res)

def visualize(y_test, y_pred, pos,name):
    tn, tp, fp, fn = get_all_stat(y_test, y_pred)

    tn = get_need_points(tn, pos)
    tp = get_need_points(tp, pos)
    fp = get_need_points(fp, pos)
    fn = get_need_points(fn, pos)
    plt.scatter(tn[:, 0], tn[:, 1], color='blue', lw=0, label='0 0')
    plt.scatter(tp[:, 0], tp[:, 1], color='green', lw=0, label='1 1')
    plt.scatter(fp[:, 0], fp[:, 1], color='red', lw=0, label='1 0')
    plt.scatter(fn[:, 0], fn[:, 1], color='pink', lw=0, label='0 1')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.title(name)
    plt.show()



def cascade(c1,w, c2, c3):
    # nn = load_nn(c1, w)
    classifier=load_classifier(c2)
    # classifier_0=load_classifier(c3)
    A,Y,indices = get_data()
    A_clean, _,_ = get_data_without_normalize()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(A, Y, indices, random_state=50,
                                                                                     test_size=0.3)
    # y_pred_nn = nn.predict_classes(X_test)
    y_pred_cl = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    # mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-9, random_state=50, n_jobs=1, verbose=1)
    # pos = mds.fit(X_test).embedding_
    #
    # visualize(y_test, y_pred_cl, pos, "adaboost")

    compare_proba(y_pred_cl, y_pred_proba, y_test)

    show_res("adaboost", y_test, y_pred_cl)
    # show_res("neural network", y_test, y_pred_nn)
    # y_res = []
    # for i in range(len(y_pred_nn)):
    #     if y_pred_nn[i] == y_pred_cl[i]:
    #         y_res.append(y_pred_nn[i])
    #     elif A_clean[indices_test[i]][1] > SPLIT:
    #         y_res.append(0)
    #     else:
    #         y_res.append(1)
    #
    # show_res("merge", y_test, y_res)


# cascade("nn_50.json","model_50.h5","ada","ada0")