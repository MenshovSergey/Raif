from sklearn import manifold
from sklearn.model_selection import train_test_split

from main import get_data
from matplotlib import pyplot as plt
import numpy as np



def main():
    A, Y, _, new_f_names = get_data()
    X_train, X_test, y_train, y_test = train_test_split(A, Y, random_state=50,test_size=0.3)
    mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-9, random_state=50,n_jobs=1, verbose=1)
    pos = mds.fit(A).embedding_

    pos_0 = []
    pos_1 = []
    for i, v in enumerate(Y):
        if v == 1:
            pos_1.append(pos[i])
        else:
            pos_0.append(pos[i])
    pos_0 = np.asarray(pos_0)
    pos_1 = np.asarray(pos_1)
    plt.scatter(pos_1[:, 0], pos_1[:, 1], color='turquoise', lw=0, label='1')
    plt.scatter(pos_0[:, 0], pos_0[:, 1], color='red', lw=0, label='0')
    print("A")
    plt.show()

main()