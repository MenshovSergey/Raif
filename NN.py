import random
import numpy as np
from keras import metrics
from keras.layers import Dense, Dropout, BatchNormalization

from keras.models import Sequential, model_from_json

from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from main import read_data, normalize_data, THRESHOLD_DATA, feature_importance, COUNT, get_FP, union_columns2, \
    prefix_for_remove2, union_columns3, factors, union_data, get_data


def baseline_model():
    # create model
    np.random.seed(42)
    model = Sequential()

    model.add(Dense(3000, input_dim=COUNT+1, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(2500, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu'))
    #model.add(Dense(325, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',metrics.binary_accuracy])
    return model

def get_data_pred():
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
        if v == 0 and random.random() < THRESHOLD_DATA:
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

    A = feature_importance(X_norm, Y, list(X))
    return A, Y, range(len(Y))

def get_FP_local(y_test, y_pred):
    res = []
    for i, (y_t, y_p) in enumerate(zip(y_test, y_pred)):
        if y_p == 0 and y_t == 1:
            res.append(i)
    return res

def test_loaded_model():
    json_file = open('nn_50.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_50.h5")
    A, Y, indices = get_data()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(A, Y, indices, random_state=50,
                                                                                     test_size=0.3)
    res = loaded_model.predict_proba(X_test)
    y_pred = loaded_model.predict_classes(X_test)
    fp = get_FP(y_test, y_pred,indices_test)
    res_f = open("result.out", "w")
    print(fp, file=res_f)
    fp = get_FP_local(y_test, y_pred)
    for i in fp:
        print (res[i])


def main():
    A, Y, indices = get_data()
    # A_pred,Y_pred,_ = get_data_pred()
    model = baseline_model()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(A, Y, indices, random_state=50,
                                                                                     test_size=0.3)
    dummy_y = np_utils.to_categorical(y_train, 2)
    dummy_y_test = np_utils.to_categorical(y_test, 2)
    model.fit(X_train, dummy_y, batch_size=50, epochs=50, validation_split=0.1)
    print(model.evaluate(X_test, dummy_y_test))
    y_pred =model.predict_classes(X_test)

    fp = get_FP(y_test,y_pred, indices_test)
    res = open("result.out","w")
    print(fp,file=res)
    res.close()

    print("\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print("F1 measure : " + str(f1_score(y_test, y_pred, average=None)))


    model_json = model.to_json()
    with open("nn_50.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_50.h5")
    print("Saved model to disk")

# main()
# test_loaded_model()