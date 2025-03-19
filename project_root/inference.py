#type:ignore
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from pydantic import conint
from pydantic import Field
from pydantic import create_model
from config import config
import pickle
import os
import numpy as np
import math
import uvicorn
from sklearn.preprocessing import PolynomialFeatures

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ENCODING_FILENAME),"rb") as file_handle:
    features_encodings = pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NOMINAL_COLUMNS_IDX_FILENAME),"rb") as file_handle:
    nominal_columns_idx = pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ORDINAL_COLUMNS_IDX_FILENAME),"rb") as file_handle:
    ordinal_columns_idx = pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NUMERIC_COLUMNS_IDX_FILENAME),"rb") as file_handle:
    numeric_columns_idx = pickle.load(file_handle)

all_columns_idx = dict()
all_columns_idx.update(nominal_columns_idx)
all_columns_idx.update(ordinal_columns_idx)
all_columns_idx.update(numeric_columns_idx)

features_min_values = dict()
features_max_values = dict()

for k,v in sorted(all_columns_idx.items()):

    if v != 2:
        features_min_values[k] = 0
        features_max_values[k] = max(list(features_encodings[k].values()))
    else:
        features_min_values[k] = 2
        features_max_values[k] = 64


def predict(X_transpose):

    nominal_columns_idx_copy = dict(zip(nominal_columns_idx.values(),nominal_columns_idx.keys()))
    nominal_features_ohe_values = list()

    for k,v in sorted(nominal_columns_idx_copy.items()):
        feature_unique_values_num = len(features_encodings[v])
        identity_mat = np.eye(feature_unique_values_num,feature_unique_values_num)
        feature_ohe_value = identity_mat[int(X_transpose[all_columns_idx[v]])]
        nominal_features_ohe_values.append(feature_ohe_value)

    X_transpose = np.concatenate((X_transpose[0:min(nominal_columns_idx.values())],
                                np.concatenate(nominal_features_ohe_values)))

    with open(os.path.join(config.SAVED_MODEL_PATH,config.SAVED_MODEL_FILE),"rb") as file_handle:
        thetas = pickle.load(file_handle)

    theta0_star = thetas[0]
    theta_star = thetas[1]
    theta_star = theta_star.reshape(theta_star.shape[0],)

    with open(os.path.join(config.SAVED_NORMALIZER_PATH,config.SAVED_NORMALIZER_FILE),"rb") as file_handle:
        normalizer = pickle.load(file_handle)

    X_transpose = X_transpose.reshape(1,-1)
    X_bar_transpose = normalizer.transform(X_transpose)
    y_hat = theta0_star + np.dot(X_bar_transpose,theta_star)

    return math.exp(y_hat.item())


app = FastAPI()

schema = {k:(int,Field(...,description=f"{k}"+" must be between {} and {}".format(features_min_values[k],
                                                                                features_max_values[k]),
                                                                                ge=features_min_values[k],
                                                                                le=features_max_values[k]))
                                                                                for k,v in all_columns_idx.items()}

InputFeatureVector = create_model("InputFeatureVector",**schema)



@app.get("/")
async def home_page():

    return "This Web App predicts the price of a used laptop in India, based on it's Type, Screen Resolution and technology, CPU, RAM, Hard Drive and technology and GPU" 
    

@app.post("/predict_price")
async def perform_prediction(input_features:InputFeatureVector):

    num_input_features = len(InputFeatureVector.model_fields.items())
    X_transpose = np.zeros((num_input_features,))

    for feature_name, field in InputFeatureVector.model_fields.items():
        X_transpose[all_columns_idx[feature_name]] = getattr(input_features,feature_name)

    y_hat = predict(X_transpose)
    return "The Price of this used laptop in India is {}".format(y_hat)



if __name__ == "__main__":
    uvicorn.run(app)
