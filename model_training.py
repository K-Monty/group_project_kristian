import pandas as pd
import numpy as np
import requests
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


url = "http://127.0.0.1:8000/"
df_bunker = []
def import_data(url):
    response = requests.get(url + "n_chunks")
    n_chunks = response.json()["chunks"]
    for i in range(n_chunks):
        response = requests.get(url + "chunk/" + str(i))
        json_text = response.text
        short = json_text[1:-1]
        short = short.replace("\\", "")
        df = pd.read_json(short, orient="records")


def train(df):
    """A model to train the data"""

    #split the dataset in train and test
    testset = df[df["year"] == 5]
    trainset = df[df["year"] == 4]

    X_train = trainset[["age_of_car_M", "Car_power_M"]].values
    X_test = testset[["age_of_car_M", "Car_power_M"]].values
    y_train = trainset["Claims1"]
    y_test = testset["Claims1"]

    # data preprocessing
    est = MinMaxScaler()
    est.fit(X_train)

    X_train = est.transform(X_train)
    X_test = est.transform(X_test)

    # train model
    clf = linear_model.TweedieRegressor()
    clf.fit(X_train, y_train)

    return [clf.coef_,
            clf.intercept_,
            clf.score(X_train, y_train),
            clf.score(X_test, y_test)]