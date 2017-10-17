from keras.models import model_from_json
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score
from NN import get_data


def load_nn(name, w):
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(w)
    return loaded_model

def load_classifier(c2):
    return pickle.load(open(c2,"rb"))


def show_res(name, y_test, y_res):
    print("\n")
    print(name+"\n")
    print(confusion_matrix(y_test, y_res))
    print("\n")
    print("F1 measure : " + str(f1_score(y_test, y_res, average=None)))

def cascade(c1,w, c2):
    nn = load_nn(c1, w)
    classifier=load_classifier(c2)
    A,Y,indices = get_data()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(A, Y, indices, random_state=50,
                                                                                     test_size=0.3)
    y_pred_nn = nn.predict_classes(X_test)
    y_pred_cl = classifier.predict(X_test)
    show_res("adaboost", y_test, y_pred_cl)
    show_res("neural network", y_test, y_pred_nn)
    y_res = []
    for i in range(len(y_pred_nn)):
        if y_pred_nn[i] == 1 or y_pred_cl[i] == 1:
            y_res.append(1)
        else:
            y_res.append(0)

    show_res("merge", y_test, y_res)


cascade("nn_50.json","model_50.h5","ada")