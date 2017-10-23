from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from main import read_data, get_data, get_data_without_normalize
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
def visualize(p_0,p_1):

    plt.scatter(p_0[:, 0], p_0[:, 1], color='blue', lw=0, label='0')
    plt.scatter(p_1[:, 0], p_1[:, 1], color='green', lw=0, label='1')

    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.show()


def show_res( y_test, y_res):
    print(confusion_matrix(y_test, y_res))
    print("\n")
    print("F1 measure : " + str(f1_score(y_test, y_res, average=None)))

def main():
    X, Y, _ = get_data_without_normalize()

    split = 21
    indices = range(len(Y))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, indices, random_state=50,
                                                                                     test_size=0.3)
    y_pred = []
    p_0 = []
    p_1 = []
    for v, y in zip(X, Y):
       if y==1:
           p_1.append((v[0],v[1]))
       else:
           p_0.append((v[0], v[1]))
    res =np.cov(X.T)
    res = []
    for i in range(len(X[0])):
        res.append([])
        for j in range(i+1, len(X[0])):
            resR = pearsonr(X[:,i],X[:,j])
            res[i].append(resR[0])

    for i in range(len(res)-1):
        print(max(res[i]))
    print(res)
    # visualize(np.asarray(p_0), np.asarray(p_1))
    # show_res(y_test, y_pred)
    #
    evcl_0 =[]
    evcl_1 =[]
    for v_y, val_evcl in zip(Y, X):
        if v_y == 0:
            evcl_0.append(val_evcl[1])
        else:
            evcl_1.append(val_evcl[1])
    c_0 = Counter(evcl_0)
    c_1 = Counter(evcl_1)
    visualize(np.asarray(list(c_0.items())), np.asarray(list(c_1.items())))
    stat_evcl = open("evcl","w")
    stat_evcl.write(str(c_0)+"\n")
    stat_evcl.write(str(c_1)+"\n")
    stat_evcl.close()

# main()