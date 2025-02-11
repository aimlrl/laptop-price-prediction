from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from pydantic import conint
from config import config
import pickle
import os
import numpy as np

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ORDINAL_COLUMNS_IDX_FILENAME),
          "rb") as file_handle:
    ordinal_columns_idx_dict = pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NOMINAL_COLUMNS_IDX_FILENAME),
          "rb") as file_handle:
    nominal_columns_idx_dict = pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NUMERIC_COLUMNS_IDX_FILENAME),
          "rb") as file_handle:
    numeric_columns_idx_dict = pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ENCODING_FILENAME),"rb") as file_handle:
    feature_values_encodings = pickle.load(file_handle)

categorical_feature_min_max_values = dict()

for feature in list(nominal_columns_idx_dict.keys()):
    feature_val_encodings = list(feature_values_encodings[feature].values())
    categorical_feature_min_max_values[feature] = conint(ge=min(feature_val_encodings),
                                                         le=max(feature_val_encodings))
    
for feature in list(ordinal_columns_idx_dict.keys()):
    feature_val_encodings = list(feature_values_encodings[feature].values())
    categorical_feature_min_max_values[feature] = conint(ge=min(feature_val_encodings),
                                                         le=max(feature_val_encodings))
    

class input_feature_vector(BaseModel):
    pass

for feature_name,datatype in list(categorical_feature_min_max_values.items()):
    input_feature_vector.__annotations__[feature_name] = datatype

app = FastAPI()

@app.get("/")
async def main_page():
    return {"Project":"A Web App to predict the price of a Second Hand Laptop in India"}