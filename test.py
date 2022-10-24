"""
Test the modules predict_api.py, logistic_regression_func.py......

"""

import pytest
from . import logistic_regression_func as my_helper_func
from . import logistic_regression_app as my_app


# TODO: test that hashing works
# TODO: test that the ratio of train and test data set is approx. as desired

def test_fetch_df():
    df = my_app.fetch_df()
