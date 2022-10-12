import os
import pickle
from typing import Union
#from ctypes import Union
from types import FunctionType

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import logistic_regression_func as my_func


def fetch_df() -> pd.DataFrame:
    data_pickle_path = r"C:\Users\redal\Code\bootcamp_ppi\python_class_with_Kristian\insurance_data.pickle"
    if not os.path.exists(data_pickle_path):
        df = my_func.InsuranceDataAPI("http://localhost:8000/").fetch_df()
        with open(data_pickle_path, 'wb') as fh:
            pickle.dump(df, fh)
    else:
        df = pickle.load(open(data_pickle_path, "rb"))
    return df


def filter_df(df: pd.DataFrame) -> Union[np.array, list]:
    # Claims1 is amount of motor claims
    # Here adding a boolean column of whether a customer makes any claims
    # TODO: preserve column names for the returned X array???
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
    
    return np.array(X), list(np.array(y).reshape(1, -1)[0])
    

def train_model_pipelines(models_to_train: list, transformer: FunctionType, scaler: FunctionType) -> list:  # return a list of pipelines
    # TODO: make pipeline adaptable to func=None, for when not all 3 steps are needed (e.g. for random forest)
    # TODO: make split_train_test_stratified SHUT UP (not keep printing train & test data)

    pipelines_obj = []
    for func in models_to_train:
        pipeline = my_func.MyMLPipeline(transformer, scaler, func).run(X_train, y_train)
        pipelines_obj.append(pipeline)
    return pipelines_obj


def evaluate_models(list_of_pipelines: list):
    for pipeline in list_of_pipelines:
        y_pred = pipeline.predict(X_test)
        print("Model: {}".format(pipeline))
        print("-----------------------")
        print("Training accuracy: {}".format(pipeline.score(X_train, y_train)))
        print("Testing accuracy: {}".format(pipeline.score(X_test, y_test)))
    
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix")
        print("----------------")
        print(cm)
    
        my_evalparam = my_func.EvalParams(cm)
        print("Recall: {}".format(my_evalparam.recall))
        print("Precision: {}".format(my_evalparam.precision))


if __name__ == "__main__":

    df = fetch_df()
    X, y = filter_df(df)

    print("Counter before undersampling: {}".format(Counter(y)))
    X_new, y_new = my_func.undersampling_nearmiss3(X, y)
    print("Counter after undersampling: {}".format(Counter(y_new)))

    X_train, y_train, X_test, y_test = my_func.split_train_test_stratified(X_new, y_new)

    # define transformer, scaler and regressor(models_to_train) used for the pipeline
    transformer = FunctionTransformer(my_func.transform_log)
    scaler = RobustScaler()

    models_to_train = [
        LogisticRegression(), 
        RandomForestClassifier(max_depth=4)
    ]

    list_of_pipelines = train_model_pipelines(models_to_train, transformer, scaler)
    evaluate_models(list_of_pipelines)

