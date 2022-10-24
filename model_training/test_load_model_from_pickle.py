import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import logistic_regression_model_training as my_model
import logistic_regression_helper_module as my_func
import constants as my_const

# can use statsmodels: https://www.statsmodels.org/stable/index.html for proper report


df = my_model.fetch_df()
X, y = my_model.filter_df(df)
X_new, y_new = my_func.undersampling_nearmiss3(X, y)
X_train, y_train, X_test, y_test = my_func.split_train_test_hash(X_new, y_new)


model_unpickled = pickle.load(open(my_const.MODEL_PICKLE_PATH, "rb"))

y_pred = model_unpickled.predict(X_test)

classification_report = classification_report(y_test, y_pred)
print(classification_report)