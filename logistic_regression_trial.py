#!/usr/bin/env python     # "here is a python program", only needed when working in unix env
# coding: utf-8           # what encoding does it use

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
# pipeline code: https://stats.stackexchange.com/questions/402470/how-can-i-use-scaling-and-log-transforming-together


class InsuranceData:
    """
    Simply a container/wrapper for insuranve_data.csv, to fetch data easier
    """
    def __init__(self, my_path):
        self.path = my_path
        self.df = pd.read_csv(r"{}".format(self.path), index_col = 0)

    def fetch(self):
        return self.df


class MyPipeline:
    """
    Simple pipeline running transformer, scaler and regressor on X_train and y_train
    Note: only use this pipeline when applying same transforner & scaler on all columns in X_train
    """
    def __init__(self, transformer, scaler, regressor):
        self.transformer = transformer
        self.scaler = scaler
        self.regressor = regressor

    def run(self, X_train, y_train):
        pipe = Pipeline(steps=[
            ('transformer', self.transformer), 
            ('scaler', self.scaler), 
            ('regressor', self.regressor)
            ], memory='sklearn_tmp_memory')
        pipe.fit(X_train, y_train)
        return pipe


class EvalParams:
    """
    Calculate evaluation parameters (recall, precision) when given a confusion matrix
    """
    def __init__(self, my_confusion_matrix):
        self.matrix = my_confusion_matrix
        self.tn, self.fp, self.fn, self.tp = self._confusion_matrix_to_conditions()
        self.recall = self._recall()
        self.precision = self._precision()

    def _confusion_matrix_to_conditions(self):
        tn, fp, fn, tp = self.matrix.ravel()
        return tn, fp, fn, tp

    def _recall(self):
        return self.tp/(self.tp + self.fp)

    def _precision(self):
        return self.tp/(self.tp + self.fn)


def correlation_test(input_df, method="pearson"):
    corr = input_df.corr(method=method)
    return corr


def nearmiss_undersampling(X, y):
    undersample = NearMiss(version=3, n_neighbors_ver3=3)
    X_new, y_new = undersample.fit_resample(X, y)
    return X_new, y_new


def stratified_train_test_split(X, y):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    return X_train, y_train, X_test, y_test


def log_transform(x):
    print(x)
    return np.log(x + 1)


if __name__ == "__main__":
    df = InsuranceData(r"C:\Users\redal\Code\bootcamp_ppi\python_class_with_Kristian\insurance_data.csv").fetch()
    any_motor_claim_boolean = df['amount_claims_motor'] > 0.0
    df["any_motor_claims"] = any_motor_claim_boolean.astype(int)

    selected_xcol = ['age_client', 'Client_Seniority', 'Car_2ndDriver_M', 'annual_payment_motor']
    selected_ycol = ['any_motor_claims']
    X = df[selected_xcol]
    y = df[selected_ycol]

    counter_before_undersampling = Counter(list(np.array(y).reshape(1, -1)[0]))
    print("Counter before undersampling: {}".format(counter_before_undersampling))

    # the original data is very imbalanced: only very few people make claims.
    # hence need to undersample first before everything else
    X_temp = np.array(X)
    y_temp = list(np.array(y).reshape(1, -1)[0])
    X_new, y_new = nearmiss_undersampling(X_temp, y_temp)
    counter_after_undersampling = Counter(y_new)
    print("Counter after undersampling: {}".format(counter_after_undersampling))

    X_train, y_train, X_test, y_test = stratified_train_test_split(X_new, y_new)

    # define transformer, scaler and regressor used for the pipeline
    transformer = FunctionTransformer(log_transform)
    scaler = RobustScaler()
    regressor = LogisticRegression()

    # run the transform -> scaling -> regressing pipeline
    pipeline = MyPipeline(transformer, scaler, regressor).run(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Training accuracy: {}".format(pipeline.score(X_train, y_train)))
    print("Testing accuracy: {}".format(pipeline.score(X_test, y_test)))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print("----------------")
    print(cm)

    my_evalparam = EvalParams(cm)
    print("Recall: {}".format(my_evalparam.recall))
    print("Precision: {}".format(my_evalparam.precision))