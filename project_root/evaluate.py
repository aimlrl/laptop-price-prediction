import pandas as pd
import numpy as np
from training.train import mse
import pickle 
import os
from config import config
from sklearn.metrics import r2_score


def evaluate_unseen_data_performance(): 

    cv_data = pd.read_csv(os.path.join(config.DATA_DIR,config.CV_DATA_FILENAME))
    X_cv_transpose = np.array(cv_data.iloc[:,0:-1])
    y_cv = np.array(cv_data.iloc[:,-1]).reshape(X_cv_transpose.shape[0],1)

    with open(os.path.join(config.SAVED_MODEL_PATH,config.SAVED_MODEL_FILE),"rb") as file_handle:
        thetas = pickle.load(file_handle)

    theta0_star = thetas[0]
    theta_star = thetas[1]

    y_cv_pred = theta0_star + np.matmul(X_cv_transpose,theta_star)
    cv_mse = np.mean((y_cv - y_cv_pred)**2)

    print("\n\nMean Squared Error on Cross Validation Data is {}".format(cv_mse))
    print("Performance on Cross Validation Data is {}".format(r2_score(y_true=y_cv,y_pred=y_cv_pred)))




if __name__ == "__main__":
    evaluate_unseen_data_performance()
