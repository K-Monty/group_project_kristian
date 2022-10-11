#!/usr/bin/env python
# coding: utf-8

# import required libraries

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from collections import Counter


# In[2]:


pd.set_option("display.max_columns", 50)


# read the dataset


df = pd.read_csv("C:/Users/M223226/Desktop/PPI/Fraud detection/transactionsdata_fraud_export.csv")

print(df.head())
print(df.info())
print(df.nunique())


# encoding the catorical columns


cat_columns = ["Locale", "Country", "OrderPackage", "TrafficType", "ActivationType", "TransactionCurrency",
               "TransactionSubtype", "CardType", "CardLevel", "CardBrand", "CardCountryIso", "Cycle"]

encod_1 = pd.get_dummies(data=df, columns=cat_columns)  
encod_2 = df["CoveryScoreMeaningfulReasons"].str.get_dummies(sep=',')
encod_3 = df["CoveryScoreReasons"].str.get_dummies(sep=',')


results = pd.concat([encod_1,encod_2, encod_3], axis = 1)
results = results.drop(columns=[
    "TransactionDate", "CardBank", "CoveryScoreReasons", "CoveryScoreMeaningfulReasons", "FADate", "FASource", "PublisherId"])


print(results.head())
print(results.info())
print(results.columns.to_list())

# correlation between not caterolical values


corr = df[["IsEurope", "NetworkId", "PublisherId", "TransactionAmount", "TransactionAmountEUR", "Is3DSecurity", "CoverScore"]].corr()

f = plt.figure(figsize=(8, 5))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
plt.show()


# In[12]:





# In[13]:


X = results.drop(columns=["IsFA"])
y = results["IsFA"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[14]:


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# In[15]:


print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")


# In[16]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[17]:


print("Accuracy of Train:", model.score(X_train, y_train))
print("Accuracy of Test:",model.score(X_test, y_test))


# In[18]:


plot_confusion_matrix(model, X_test, y_test) 
plt.show()






