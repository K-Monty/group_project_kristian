#!/usr/bin/env python
# coding: utf-8

import os
import requests
import re
import time
import string
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, plot_confusion_matrix

#from sklearn.pipeline import make_pipeline
#from sklearn.pipeline import Pipeline

# Oversample and plot imbalanced dataset with SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

raw_df = pd.read_csv("../datasets/data_mendeley.csv")
raw_df.columns = raw_df.columns.str.lower()

raw_df.rename(columns = {'age_of_car_m':'car_age', 'car_power_m':'car_power', 'car_2nddriver_m':'second_driver'},inplace=True)
raw_df.drop(['unnamed: 0','num_policiesc','polid'],axis=1,inplace=True)

to_drop = ['types','policy_paymentmethodh','nclaims2','claims2','insuredcapital_continent_re','insuredcapital_content_re']
raw_df = raw_df.drop(to_drop,axis = 1)
def add_col(row):
    if row['nclaims1'] > 0:
        val = 1
    else:
        val = 0
    return val

raw_df['nclaims_boolean'] = raw_df.apply(add_col, axis=1)

from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer

column_trans = ColumnTransformer(
    [
        (
            "binned_numeric", KBinsDiscretizer(n_bins=10), 
            ["client_seniority", "age_client","car_power"]),
        (
            "passthrough_numeric", "passthrough", 
            ["client_seniority", "car_power"])
    ],
    remainder="drop",
)

target = ["nclaims_boolean"]
drop_columns = ["year", "nclaims1", "claims1"]

df_train_raw = raw_df[raw_df.year == 4].drop(drop_columns, axis=1)
df_test_raw  = raw_df[raw_df.year == 5].drop(drop_columns, axis=1)

df_train_raw["nclaims_boolean"] = df_train_raw["nclaims_boolean"].astype(bool)
df_test_raw["nclaims_boolean"] = df_test_raw["nclaims_boolean"].astype(bool)

X_train_raw = column_trans.fit_transform(df_train_raw.drop(target, axis=1))
X_test_raw = column_trans.fit_transform(df_test_raw.drop(target, axis=1))

y_train_raw = df_train_raw[target]
y_test_raw = df_test_raw[target]

#column_trans.get_feature_names_out()

logr_raw = LogisticRegression(max_iter=1000)
logr_raw.fit(X_train_raw, np.ravel(y_train_raw))


#raw_predictions = logr_raw.predict(X_test_raw)
#confusion_matrix(y_test_raw, raw_predictions)
#precision_score(y_test_raw, raw_predictions).round(5)

smote = SMOTE(random_state = 101)
X_train_over, y_train_over = smote.fit_resample(X_train_raw, y_train_raw)

logr_over = LogisticRegression(max_iter=1000)
logr_over.fit(X_train_over, np.ravel(y_train_over))


over_predictions = logr_over.predict(X_test_raw)
cm_over = confusion_matrix(y_test_raw, over_predictions)

precision_score(y_test_raw, over_predictions).round(5)




