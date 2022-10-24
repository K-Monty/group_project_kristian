import os
import pickle
from typing import Union
from types import FunctionType

import numpy as np
import pandas as pd
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import logistic_regression_helper_module as my_func
import constants as my_const

# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time


# TODO: detect if the data cached in data_pickle path has same nchunks as in the api
def fetch_df(rewrite=False) -> pd.DataFrame:
    data_pickle_path = my_const.DATA_PICKLE_PATH
    if not os.path.exists(data_pickle_path) or rewrite == True:
        df = my_func.InsuranceDataAPI("http://localhost:8000/").fetch_df()
        with open(data_pickle_path, 'wb') as fh:
            pickle.dump(df, fh)
    else:
        df = pickle.load(open(data_pickle_path, "rb"))
    return df


def filter_df(df: pd.DataFrame) -> Union[pd.DataFrame, list]:
    # Claims1 is amount of motor claims
    # Here adding a boolean column of whether a customer makes any claims
    any_motor_claim_boolean = df["Claims1"] > 0.0
    df["any_motor_claims"] = any_motor_claim_boolean.astype(int)

    selected_xcol = [
        "Age_client",
        "Client_Seniority",
        "Car_2ndDriver_M",
        "Policy_PaymentMethodA",  # "annual_payment_motor",
    ]
    selected_ycol = ["any_motor_claims"]
    
    X = df[selected_xcol]
    y = df[selected_ycol]
    
    return X, list(np.array(y).reshape(1, -1)[0])


def train_model_pipeline(X_train: pd.DataFrame, y_train: list, transformer: FunctionType, scaler: FunctionType, regressor: FunctionType) -> Pipeline:
    pipeline = my_func.MyMLPipeline(transformer, scaler, regressor).run(X_train, y_train)
    return pipeline


def save_model_as_pickle_object(model: Pipeline):
    pickle.dump(model, open("model.pkl", "wb"))


def evaluate_model(X_test: pd.DataFrame, y_test: list, model: Pipeline):
    y_pred = model.predict(X_test)
    print("Confusion Matrix")
    print("----------------")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    df = fetch_df()
    X, y = filter_df(df)
    X_new, y_new = my_func.undersampling_nearmiss3(X, y)

    print("Counter before undersampling: {}".format(Counter(y)))
    print("Counter after undersampling: {}".format(Counter(y_new)))

    X_train, y_train, X_test, y_test = my_func.split_train_test_hash(X_new, y_new)

    transformer = FunctionTransformer(my_func.transform_log)
    scaler = RobustScaler()
    regressor = LogisticRegression()

    model = my_func.MyMLPipeline(transformer, scaler, regressor).run(X_train, y_train)

    pickle.dump(model, open(my_const.MODEL_PICKLE_PATH, "wb"))

    evaluate_model(X_test, y_test, model)

