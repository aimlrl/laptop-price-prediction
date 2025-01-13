from data_transformation.feature_transformation import feature_transformation_pipeline
from data_transformation.feature_engineering import engineered_feaures,normalize_data
from config import config



def complete_pipeline():

    cleaned_transformed_data = feature_transformation_pipeline()
    X_transpose_engineered,y = engineered_feaures(cleaned_transformed_data,config.DEGREE,
                                                  config.IS_ONLY_INTERACTION)
    X_bar_transposed_engineered = normalize_data(X_transpose_engineered)

    X_train_transpose = X_bar_transposed_engineered[0:int(config.TRAINING_DATA_FRAC*X_bar_transposed_engineered.shape[0])]
    y_train = y[0:int(config.TRAINING_DATA_FRAC*y.shape[0])]

    X_cv_transpose = X_bar_transposed_engineered[int(config.TRAINING_DATA_FRAC*X_bar_transposed_engineered.shape[0]):int((config.TRAINING_DATA_FRAC+config.CV_DATA_FRAC)*X_bar_transposed_engineered.shape[0])]
    y_cv = y[int(config.TRAINING_DATA_FRAC*y.shape[0]):int((config.TRAINING_DATA_FRAC+config.CV_DATA_FRAC)*y.shape[0])]

    X_test_transpose = X_bar_transposed_engineered[int((config.TRAINING_DATA_FRAC+config.CV_DATA_FRAC)*X_bar_transposed_engineered.shape[0]):]
    y_test = y[int((config.TRAINING_DATA_FRAC+config.CV_DATA_FRAC)*y.shape[0]):]

    return (X_train_transpose,y_train), (X_cv_transpose,y_cv), (X_test_transpose,y_test)

    