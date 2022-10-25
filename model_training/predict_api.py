#!/usr/bin/env python3

"""
run it with:
    uvicorn prediction_server:app --reload
"""
import pickle
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import pandas as pd
from pydantic import BaseModel

import constants as my_const


# https://stackoverflow.com/questions/71845425/fastapi-array-of-json-in-request-body-for-machine-learning-prediction
# TODO: convert yes or no into 0. or 1. before feed into /predict
# TODO: raw data needs to go through same transform etc


# intialize web server
app = FastAPI()


class Item(BaseModel):
    age: int
    client_seniority: float
    second_driver: int
    annual_payment_motor: int


@app.post("/predict")  # URL suffix, URL path or endpoint
def predict(item: Item):
    """uses the ML model"""
    model = pickle.load(open(my_const.MODEL_PICKLE_PATH, "rb"))
    item = pd.DataFrame([jsonable_encoder(item)])
    print(item)
    # TODO: transform item!
    # user_input_data = np.array([item.age, item.client_seniority, item.second_driver, item.annual_payment_motor])
    result = model.predict(item).tolist()[0]
    return {"Model Prediction": result}
