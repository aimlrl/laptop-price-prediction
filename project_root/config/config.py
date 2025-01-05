ROOT_DIR_PATH = "~/AiML-projects/laptop-price-prediction/project_root"
FILENAME = "laptopData.csv"
DATA_DIR = "dataset"



CATEGORICAL_COLUMNS = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
NUMERIC_COLUMNS = ['Weight', 'Inches', 'Ram']
ORDINAL_COLUMNS = CATEGORICAL_COLUMNS[0:2]
NOMINAL_COLUMNS = CATEGORICAL_COLUMNS[2:]
TARGET_COLUMN = "Price"



REFINED_COLUMNS = ['TypeName', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu']



SAVED_ENCODINGS_PATH = "~/AiML-projects/laptop-price-prediction/project_root/encodings"
ENCODING_FILENAME = "columns_label_encodings.pkl"


COLUMNS_TO_CLEAN = ["Ram", "Weight", "Inches"]


DEGREE = 2


IS_ONLY_INTERACTION = False


TRAINING_DATA_FRAC = 0.7
CV_DATA_FRAC = 0.2
TESTING_DATA_FRAC = 0.1


EPSILON = 10**(-4)
TOLERANCE = 10**(-5)


SAVED_MODEL_FILE = "trained_model.pkl"
SAVED_MODEL_PATH = "~/AiML-projects/laptop-price-prediction/project_root/models"