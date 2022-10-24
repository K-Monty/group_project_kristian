import mlflow

from copyreg import pickle
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel

import logistic_regression_app as my_app
import logistic_regression_func as my_func


# https://mlflow.org/docs/latest/quickstart.html

# context manager (pygame, write text files -- with open(file, read-or-write) as sth: ...)
# can wrap this with statement into a function (model_saving_path as input_param), and return the model

# can start_run param set model name into sth nicer?
# name + timestamp (so wont accidentally overwrite it later!)
with mlflow.start_run():
    df = my_app.fetch_df()
    
    X, y = my_app.filter_df(df)

    X_new, y_new = my_func.undersampling_nearmiss3(X, y)

    X_train, y_train, X_test, y_test = my_func.split_train_test_hash(X_new, y_new)


    trees = 77
    mlflow.log_param("n_estimators", trees)  # log a hyperparameters

    mlflow.log_metric("How stupid are you", 100)

    model = LogisticRegression()  # or similar
    model.fit(X_train, y_train)

    mlflow.log_metric("training score", model.score(X_train, y_train))
    mlflow.log_metric("test score", model.score(X_test, y_test))

    

    mlflow.sklearn.log_model(model, "model")