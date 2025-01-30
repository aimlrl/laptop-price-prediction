from data_loading.data_loading import load_data
from data_cleaning.data_cleaning import drop_rows_columns, clean_columns
from data_transformation.data_conversion import label_encode_columns, convert_nominal_to_ohe
from data_transformation.data_conversion import save_column_label_encodings, save_nominal_columns_idx
from data_transformation.data_conversion import save_ordinal_columns_idx
from config import config
import pandas as pd


def feature_transformation_pipeline():

    data = load_data()
    reduced_data = drop_rows_columns(data)
    cleaned_data = clean_columns(reduced_data)

    column_label_encodings, transformed_data = label_encode_columns(cleaned_data)
    save_column_label_encodings(column_label_encodings)
    
    cleaned_transformed_data = pd.concat([transformed_data[config.REFINED_COLUMNS],transformed_data[config.TARGET_COLUMN]],axis=1)
    present_nominal_columns = list(set(config.REFINED_COLUMNS).intersection(set(config.NOMINAL_COLUMNS)))
    save_nominal_columns_idx(present_nominal_columns)
    
    ohe_df = convert_nominal_to_ohe(cleaned_transformed_data,present_nominal_columns)
    cleaned_transformed_data.drop(labels=present_nominal_columns,axis=1,inplace=True)
    save_ordinal_columns_idx(cleaned_transformed_data.columns)

    cleaned_transformed_data = pd.concat([cleaned_transformed_data,ohe_df],axis=1)

    return cleaned_transformed_data



