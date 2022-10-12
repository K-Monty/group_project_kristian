#!/usr/bin/env python     # "here is a python program", only needed when working in unix env
# coding: utf-8           # what encoding does it use

"""
This module fetches insurance data from a fastapi (need to run api server before this script!) 
and run simple LogisticRegression pipeline on the data.
Result indicates whether a customer will make a motor claim (0 or 1).
"""

# Module level dunder names
__author__ = "K-Monty"
__license__ = "MIT"
__version__ = "0.1.0"
__email__ = "kmgoh1995@gmail.com"

import json
from types import FunctionType
import requests
import numpy as np
import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt

from typing import Union
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# fastapi path: C:\Users\redal\Code\bootcamp_ppi\python_class_with_Kristian\kristians_github_cloned\ml2product\data_source
# https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
# pipeline code: https://stats.stackexchange.com/questions/402470/how-can-i-use-scaling-and-log-transforming-together


class InsuranceDataAPI:
    """
    A container/wrapper to fetch data from fastapi.
    NEED TO RUN API SERVER FIRST BEFORE RUNNING THIS!!!

    Attributes
    ----------
    self.url_main: http path where the data server is

    Returns
    -------
    mega_df: <pandas.core.frame.dataframe>
        pandas dataframe of all data collected from fastapi
    """

    def __init__(self, fastapi_url: str):
        self.url_main = fastapi_url

    def fetch_df(self) -> pd.DataFrame:
        nchunks = requests.get(self.url_main + "n_chunks").json()["chunks"]
        mega_json = []

        for i in range(nchunks):
            chunk_data_url = self.url_main + "chunk/" + str(i)
            response = requests.get(chunk_data_url)
            json_text = response.text
            json_text_short = json_text[1:-1]
            json_text_short = json_text_short.replace("\\", "")
            # convert string into json
            json_obj = json.loads(json_text_short)
            mega_json.extend(json_obj)

        mega_df = pd.DataFrame.from_records(mega_json)
        return mega_df


class MyMLPipeline:
    """
    Simple pipeline running transformer, scaler and regressor on input X_train and y_train
    Note: only use this pipeline when applying same transformer, scaler and regressor
    on all columns in X_train

    Parameters
    ----------
    transformer: function
    scaler: function
    regressor: function
    """

    def __init__(
        self, transformer: FunctionType, scaler: FunctionType, regressor: FunctionType
    ):
        self.transformer = transformer
        self.scaler = scaler
        self.regressor = regressor

    def run(self, X_train: np.array, y_train: np.array) -> Pipeline:
        pipe = Pipeline(
            steps=[
                ("transformer", self.transformer),
                ("scaler", self.scaler),
                ("regressor", self.regressor),
            ],
            memory="sklearn_tmp_memory",
        )
        pipe.fit(X_train, y_train)
        return pipe


class EvalParams:
    """
    Calculate evaluation parameters (recall, precision) when given a confusion matrix

    Parameters
    ----------
    my_confusion_matrix: np.array
        (2 x 2) confusion matrix

    Attributes
    ----------
    self.tn, self.fp, self.fn, self.tp: integers
    self.recall: integer
    self.precision: integer
    """

    def __init__(self, my_confusion_matrix: np.array):
        self.matrix = my_confusion_matrix
        self.tn, self.fp, self.fn, self.tp = self._confusion_matrix_to_conditions()
        self.recall = self._recall()
        self.precision = self._precision()

    def _confusion_matrix_to_conditions(self) -> int:
        tn, fp, fn, tp = self.matrix.ravel()
        return tn, fp, fn, tp

    def _recall(self) -> int:
        return self.tp / (self.tp + self.fp)

    def _precision(self) -> int:
        return self.tp / (self.tp + self.fn)


def run_correlation_test(input_df: pd.DataFrame, method="pearson") -> pd.DataFrame:
    corr = input_df.corr(method=method)
    return corr


def undersampling_nearmiss3(X: np.array, y: list) -> Union[np.array, list]:
    undersample = NearMiss(version=3, n_neighbors_ver3=3)
    X_new, y_new = undersample.fit_resample(X, y)
    return X_new, y_new


def split_train_test_stratified(
    X: np.array, y: list
) -> Union[np.array, list, np.array, list]:
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    sss.get_n_splits(X, y)

    for train_index, test_index in sss.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    return X_train, y_train, X_test, y_test


def transform_log(x: int) -> int:
    print(x)
    return np.log(x + 1)