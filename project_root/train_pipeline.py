from transform_pipeline import complete_pipeline
from training.train import training
from config import config
import os
import pickle


def run_training():

    training_data, cv_data, testing_data = complete_pipeline()
    trained_params = training(config.EPSILON,training_data[0],
                              training_data[1],config.TOLERANCE)
    
    with open(os.path.join(config.SAVED_MODEL_PATH,config.SAVED_MODEL_FILE),"wb") as file_handle:
        pickle.dump(trained_params,file_handle)



if __name__ == "__main__":
    run_training()


