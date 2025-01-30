from transform_pipeline import complete_pipeline
from training.train import training
from config import config
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def run_training():

    if len(os.listdir(config.DATA_DIR)) == 1:
        training_data, cv_data, testing_data = complete_pipeline()                                       

        X_train_df = pd.DataFrame(training_data[0])
        y_train_df = pd.DataFrame(training_data[1])

        X_cv_df = pd.DataFrame(cv_data[0])
        y_cv_df = pd.DataFrame(cv_data[1])

        X_test_df = pd.DataFrame(testing_data[0])
        y_test_df = pd.DataFrame(testing_data[1])

        training_data = pd.concat([X_train_df,y_train_df],axis=1)
        training_data.to_csv(os.path.join(config.DATA_DIR,config.TRAINING_DATA_FILENAME),
                                                   index=False)
        
        cv_data = pd.concat([X_cv_df,y_cv_df],axis=1)
        cv_data.to_csv(os.path.join(config.DATA_DIR,config.CV_DATA_FILENAME),
                                             index=False)
        
        testing_data = pd.concat([X_test_df,y_test_df],axis=1)
        testing_data.to_csv(os.path.join(config.DATA_DIR,config.TESTING_DATA_FILENAME),
                                                 index=False)
        
    training_data = pd.read_csv(os.path.join(config.DATA_DIR,config.TRAINING_DATA_FILENAME))
    X_train_transpose = np.array(training_data.iloc[:,0:-1])
    y_train = np.array(training_data.iloc[:,-1]).reshape(X_train_transpose.shape[0],1)
    
    trained_params = training(config.EPSILON,X_train_transpose,y_train,config.TOLERANCE)
    
    with open(os.path.join(config.SAVED_MODEL_PATH,config.SAVED_MODEL_FILE),"wb") as file_handle:
        pickle.dump(trained_params,file_handle)



if __name__ == "__main__":
    run_training()


