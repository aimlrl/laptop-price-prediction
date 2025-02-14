from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy.stats as s
from config import config
import os
import pickle



def engineered_feaures(data,features_degree,is_only_interaction):

    nth_degree_feature_engineer = PolynomialFeatures(degree=features_degree,interaction_only=is_only_interaction)
    X_transpose = np.array(data.drop(labels=config.TARGET_COLUMN,axis=1))
    y = np.array(data[config.TARGET_COLUMN]).reshape(X_transpose.shape[0],1)

    if s.moment(a=y,order=3) != 0:
        y = np.log(y)

    if features_degree > 1:
        X_transpose_engineered = nth_degree_feature_engineer.fit_transform(X_transpose)
    else:
        X_transpose_engineered = X_transpose

    return X_transpose_engineered,y




def normalize_data(X_transpose_engineered):

    zero_mean_one_std_scaler = StandardScaler()
    zero_mean_one_std_scaler.fit(X_transpose_engineered)
    X_bar_transpose_engineered = zero_mean_one_std_scaler.transform(X_transpose_engineered)

    with open(os.path.join(config.SAVED_NORMALIZER_PATH,config.SAVED_NORMALIZER_FILE),"wb") as file_handle:
        pickle.dump(zero_mean_one_std_scaler,file_handle)

    return X_bar_transpose_engineered