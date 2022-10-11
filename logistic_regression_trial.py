#!/usr/bin/env python     # "here is a python program", only needed when working in unix env
# coding: utf-8           # what encoding does it use

# 'make' is a unix programme, nothing to do with python
# PEP8: python standard library, one line empty, then pip installed modules
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


# class df, function undersample, function stratified split, function pipeline

class InsuranceData:
    def __init__(self, my_path):
        self.path = my_path
        self.df = pd.read_csv(r"{}".format(self.path), index_col = 0)

    def fetch(self):
        return self.df



df = pd.read_csv(r"C:\Users\redal\Code\bootcamp_ppi\python_class_with_Kristian\insurance_data.csv", index_col = 0)

my_data = InsuranceData(r"C:\Users\redal\Code\bootcamp_ppi\python_class_with_Kristian\insurance_data.csv").fetch()

print(my_data)

exit()
# add one more column in the 
m = df['amount_claims_motor'] > 0.0
df["any_motor_claims"] = m.astype(int)

# columns with >2% colinearity with the *amount_claims_motor* (and are not 'futuristic')
# use this for future anaysis
selected_xcol = ['age_client', 'Client_Seniority', 'Car_2ndDriver_M', 'annual_payment_motor']
selected_ycol = ['any_motor_claims']

X = df[selected_xcol]
y = df[selected_ycol]

counter_before_undersampling = Counter(list(np.array(y).reshape(1, -1)[0]))
print("Counter before undersampling: {}".format(counter_before_undersampling))

X_temp = np.array(X)
y_temp = list(np.array(y).reshape(1, -1)[0])

undersample = NearMiss(version=3, n_neighbors_ver3=3)

X_new, y_new = undersample.fit_resample(X_temp, y_temp)
counter_after_undersampling = Counter(y_new)
print("Counter after undersampling: {}".format(counter_after_undersampling))

# training of logistic regression starts here...
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
sss.get_n_splits(X_new, y_new)

for train_index, test_index in sss.split(X_new, y_new):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = np.array(y_new)[train_index], np.array(y_new)[test_index]


def log_transform(x):
    print(x)
    return np.log(x + 1)

transformer = FunctionTransformer(log_transform)
scaler = RobustScaler()
regressor = LogisticRegression()

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

pipe = Pipeline(steps=[('transformer', transformer), ('scaler', scaler), ('regressor', regressor)], memory='sklearn_tmp_memory')

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
pipe.score(X_test, y_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)

tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fp)
precision = tp / (tp + fn)

print("Recall: {}".format(recall))
print("Precision: {}".format(precision))


