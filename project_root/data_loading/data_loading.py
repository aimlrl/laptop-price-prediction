import pandas as pd
import os
from config import config

def load_data():
    dataset_dir_path = os.path.join(config.ROOT_DIR_PATH,config.DATA_DIR)
    data_path = os.path.join(dataset_dir_path,config.FILENAME)
    data = pd.read_csv(data_path)
    return data
