ROOT_DIR_PATH = "~/AiML-projects/laptop-price-prediction/project_root"
FILENAME = "laptopData.csv"
DATA_DIR = "dataset"



CATEGORICAL_COLUMNS = ["Company", "TypeName", "ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys"]
NUMERIC_COLUMNS = ["Weight", "Inches", "Ram"]
NOMINAL_COLUMNS = CATEGORICAL_COLUMNS[0:2]
ORDINAL_COLUMNS = CATEGORICAL_COLUMNS[2:]
TARGET_COLUMN = "Price"



REFINED_COLUMNS = ["TypeName", "ScreenResolution", "Cpu", "Ram", "Memory", "Gpu"]



SAVED_ENCODINGS_PATH = "encodings"
ENCODING_FILENAME = "columns_label_encodings.pkl"
NOMINAL_COLUMNS_IDX_FILENAME = "nominal_columns_idx.pkl"
ORDINAL_COLUMNS_IDX_FILENAME = "ordinal_columns_idx.pkl"
NUMERIC_COLUMNS_IDX_FILENAME = "numeric_columns_idx.pkl"


COLUMNS_TO_CLEAN = ["Ram", "Weight", "Inches"]


DEGREE = 1


IS_ONLY_INTERACTION = False


TRAINING_DATA_FRAC = 0.7
CV_DATA_FRAC = 0.2
TESTING_DATA_FRAC = 0.1


EPSILON = 10**(-2)
TOLERANCE = 10**(-5)



TRAINING_DATA_FILENAME = "training_data.csv"
CV_DATA_FILENAME = "cv_data.csv"
TESTING_DATA_FILENAME = "testing_data.csv"


SAVED_MODEL_FILE = "trained_model.pkl"
SAVED_MODEL_PATH = "models"

SAVED_NORMALIZER_FILE = "trained_normalizer.pkl"
SAVED_NORMALIZER_PATH = "data_transformation"


PENALTY = "l2"
LAMBDA = 5
