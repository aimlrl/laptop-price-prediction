from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel,conint
from config import config
import pickle
import os
import numpy as np


with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ORDINAL_COLUMNS_IDX_FILENAME),"rb") as file_handle:
    ordinal_columns_idx_dict = pickle.load(file_handle)

ordinal_columns_idx_dict_keys = list(ordinal_columns_idx_dict.keys())
ordinal_columns_idx_dict_values = list(ordinal_columns_idx_dict.values())
ordinal_columns_idx_dict = dict(zip(ordinal_columns_idx_dict_values,ordinal_columns_idx_dict_values))

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NOMINAL_COLUMNS_IDX_FILENAME),"rb") as file_handle:
    nominal_columns_idx_dict = pickle.load(file_handle)

nominal_columns_idx_dict_keys = list(nominal_columns_idx_dict.keys())
nominal_columns_idx_dict_values = np.array(list(nominal_columns_idx_dict.values())) + len(ordinal_columns_idx_dict)
nominal_columns_idx_dict_values = list(nominal_columns_idx_dict_values)
nominal_columns_idx_dict = dict(zip(nominal_columns_idx_dict_values,nominal_columns_idx_dict_keys))

all_columns_idx_dict = dict()
all_columns_idx_dict.update(ordinal_columns_idx_dict)
all_columns_idx_dict.update(nominal_columns_idx_dict)
all_columns_idx_dict = sorted(all_columns_idx_dict)



with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ENCODING_FILENAME),"rb") as file_handle:
    feature_values_encodings = pickle.load(file_handle)

feature_min_max_values = dict()

for feature in list(feature_values_encodings.keys()):
    feature_val_encodings = list(feature_values_encodings[feature].values())
    feature_min_max_values[feature] = conint(ge=min(feature_val_encodings),le=max(feature_val_encodings))

features = tuple()
values_type = tuple() 
for feature in list(all_columns_idx_dict.values()):
    features = features + (feature,)
    values_type = values_type + (feature_min_max_values[feature],)

features_values_type_list = list(zip(features,values_type))



class input_feature_vector(BaseModel):
    pass

for attr_name,attr_type in features_values_type_list:
    input_feature_vector.__annotations__[attr_name] = attr_type




app = FastAPI()


