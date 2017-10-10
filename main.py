import pandas as pd
import sklearn.preprocessing as preproccesing
cat_coulmns = ["Event_type", "Insurer_type", "Owner_region", "Owner_type", "Sales_channel", "VEH_aim_use", "VEH_model",
               "VEH_type_name"]
drop_columns = ["bad","claim_id"]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def read_data():
    data = pd.read_csv("data/all.csv", sep=';', decimal=",")
    data = pd.get_dummies(data, columns=cat_coulmns)
    y = data.get("bad")
    data = data.drop(drop_columns,1)
    data = data.fillna(method='pad')
    return data, y

def normalize_data(data):
    scaler = preproccesing.MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def main():
    X, Y = read_data()
    X_norm = normalize_data(X)
    rf = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X_norm,Y)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))
main()