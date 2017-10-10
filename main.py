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

cat_coulmns = ["Event_type", "Insurer_type", "Owner_region", "Owner_type", "Sales_channel", "VEH_aim_use", "VEH_model",
               "VEH_type_name"]
drop_columns = ["bad", "claim_id"]
names = ["RandomForest", "GBM", "AdaBoost", "SVM", "Dummy"]
classifiers = [RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42),
               AdaBoostClassifier(random_state=42),
               svm.SVC(random_state=42), DummyClassifier(random_state=42)]

THRESHOLD_VEH_MODEL = 10

NEW_VEH_MODEL_NAME = "VEH_model_another"


def read_data():
    data = pd.read_csv("data/all.csv", sep=';', decimal=",")
    data = pd.get_dummies(data, columns=cat_coulmns)
    y = data.get("bad")
    data = data.drop(drop_columns, 1)
    data = data.fillna(method='pad')
    small_veh_models=[]
    another = [0] * len(data.values)
    for name_column in list(data):
        if "VEH_model" in name_column:
            values = list(data.get(name_column))
            count = Counter(values)[1]
            if count < THRESHOLD_VEH_MODEL:
                small_veh_models.append(name_column)
                need_index = [i for i, x in enumerate(values) if x == 1]
                for i in need_index:
                    another[i] = 1
    data = data.drop(small_veh_models, 1)
    data[NEW_VEH_MODEL_NAME] = another
    return data, y


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


def compare_classifiers(X, Y, names, classifiers):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=50)

    # sklearn_pca = PCA(n_components=230)
    # X_train = sklearn_pca.fit_transform(X_train)
    # X_test = sklearn_pca.transform(X_test)

    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(name)
        show_metrics(y_test, y_pred)
        print()

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
    X, Y = read_data()
    X_norm = normalize_data(X)
    compare_classifiers(X_norm, Y, names, classifiers)
    feature_importance(X_norm, Y, list(X))
    pca(X_norm)

main()
