from keras.models import model_from_json
import pickle

from sklearn import manifold
from sklearn.ensemble import AdaBoostClassifier, IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score
from NN import get_data
from matplotlib import pyplot as plt
import numpy as np

from main import get_data_without_normalize, compare_classifiers, classifiers, names

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

def find_ouliers(pos):
    clf = IsolationForest(random_state=42)
    clf.fit(pos)
    y_pred = clf.predict(pos)
    # xx, yy = np.meshgrid(np.linspace(-450, 200, 50), np.linspace(-250, 200, 50))
    xx, yy = np.meshgrid(np.linspace(-450, 1500, 50), np.linspace(-450, 1100, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    y_norm = []
    y_outlier = []
    index_outlier = []
    for i, v in enumerate(y_pred):
        if v == 1:
            y_norm.append(pos[i])
        else:
            y_outlier.append(pos[i])
            index_outlier.append(i)

    y_norm = np.asarray(y_norm)
    y_outlier = np.asarray(y_outlier)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.scatter(y_norm[:, 0], y_norm[:, 1], color='blue', lw=0, label='NORM')
    plt.scatter(y_outlier[:, 0], y_outlier[:, 1], color='red', lw=0, label='Outlier')


    plt.show()
    return index_outlier

def draw_nn(c1,w):
    nn = load_nn(c1,w)
    from keras.utils import plot_model
    plot_model(nn, to_file='model.png', show_shapes=True)

def cascade(c1,w, c2, c3):
    # nn = load_nn(c1, w)
    classifier=load_classifier(c2)
    # classifier_0=load_classifier(c3)
    # A,Y,indices = get_data()
    A_clean, Y,_ = get_data_without_normalize()
    X_train, X_test, y_train, y_test = train_test_split(A_clean, Y, random_state=50,
                                                                                     test_size=0.3)
    mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-9, random_state=50, n_jobs=1, verbose=1)
    pos = mds.fit(A_clean).embedding_

    index_outliers = find_ouliers(pos)
    # index_outliers = [0, 18, 19, 37, 42, 46, 54, 57, 67, 68, 69, 81, 86, 90, 91, 92, 97, 101, 120, 129, 155, 203, 204, 205, 221, 224, 236, 238, 256, 264, 286, 299, 308, 329, 330, 342, 343, 361, 362, 363, 369, 372, 386, 401, 455, 469, 473, 487, 502, 520, 526, 532, 536, 542, 550, 551, 559, 563, 564, 578, 580, 591, 606, 607, 608, 612, 617, 628, 629, 650, 654, 666, 667, 668, 675, 678, 690, 705, 719, 728, 729, 730, 731, 734, 771, 785, 787, 791, 792, 805, 837, 845, 851, 868, 879, 896, 909, 921, 925, 938, 948, 954, 960, 961, 990, 991, 1079, 1108, 1118, 1122, 1129, 1130, 1136, 1162, 1175, 1196, 1210, 1223, 1240, 1241, 1244, 1250, 1253, 1262, 1275, 1276, 1288, 1301, 1304, 1324, 1325, 1339, 1356, 1375, 1380, 1389, 1393, 1394, 1402, 1403, 1404, 1405, 1436, 1437, 1438, 1454, 1455, 1457, 1468, 1494, 1501, 1509]

    # print(index_outliers)
    #
    # A_clean = np.delete(A_clean, index_outliers, 0)
    # Y = np.delete(Y, index_outliers, 0)
    #
    #
    #
    # # y_pred_nn = nn.predict_classes(X_test)
    # # ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    # # ada.fit(X_train, y_train)
    # # y_pred_cl = ada.predict(X_test)
    # compare_classifiers(A_clean, Y, names, classifiers)
    #
    # visualize(y_test, y_pred_cl, pos, "adaboost")
    #
    # compare_proba(y_pred_cl, y_pred_proba, y_test)
    #
    # show_res("adaboost", y_test, y_pred_cl)
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


cascade("nn_50.json","model_50.h5","ada","ada0")
# draw_nn("nn_50.json","model_50.h5")