"""
run it with:
    uvicorn prediction_server:app --reload
"""

from fastapi import FastAPI
import numpy as np
import model_training
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


# intialize web server
app = FastAPI()


@app.post("/data_frame")   # URL suffix, URL path or endpoint
def data_frame():
    """import the dataset"""
    df = model_training.import_data("http://127.0.0.1:8000/")
    model = model_training.train(df)
    json_compatible_item_data = jsonable_encoder(model)
    return JSONResponse(content=json_compatible_item_data)




#class Item(BaseModel):
 #   gender: bool
 #   age: int
  #  income: float


#@app.post("/predict")
#def predict(item: Item):
#    """uses the ML model"""
#    df = [item]
#    result = model_dummy.get_prediction(df)
#    return {"probability": result}