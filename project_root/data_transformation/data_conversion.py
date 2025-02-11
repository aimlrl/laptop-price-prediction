from config import config
import pandas as pd
import pickle
import os
import numpy as np



def label_encode_columns(data):

    columns_label_encodings = dict()
    for column_name in config.CATEGORICAL_COLUMNS:

        d = dict()
        for value in data[column_name].unique():
            d[value] = data[data[column_name] == value][config.TARGET_COLUMN].mean()

        input_target_df = pd.DataFrame(data={column_name:d.keys(),f"Mean {config.TARGET_COLUMN}":d.values()})
        input_target_df.sort_values(by=f"Mean {config.TARGET_COLUMN}",inplace=True)
        input_target_df.reset_index(inplace=True,drop=True)

        data[column_name] = data[column_name].replace(to_replace=list(input_target_df[column_name]),
                              value=list(input_target_df.index)).infer_objects(copy=False)

        columns_label_encodings[column_name] = dict(zip(list(input_target_df[column_name]),list(input_target_df.index)))

    return columns_label_encodings, data



def convert_nominal_to_ohe(data,present_nominal_columns):

    ohe_columns_dfs = list()
    for column in config.REFINED_COLUMNS:

        if column in present_nominal_columns:

            column_unique_vals = data[column].unique().shape[0]
            identity_mat = np.eye(column_unique_vals,column_unique_vals)
            ohe_converted_column = identity_mat[data[column]]
            ohe_converted_column_df = pd.DataFrame(data=ohe_converted_column,
                                               columns=[column+f"_{i}" for i in range(column_unique_vals)])
            ohe_columns_dfs.append(ohe_converted_column_df)

    ohe_df = pd.concat(ohe_columns_dfs,axis=1)

    return ohe_df
        


def save_column_label_encodings(columns_label_encodings):

    with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ENCODING_FILENAME),"wb") as file_handle:
        pickle.dump(columns_label_encodings,file_handle)



def save_nominal_columns_idx(present_nominal_columns):

    present_nominal_columns_idx_dict = dict()
    remaining_columns = [column for column in config.REFINED_COLUMNS if column not in config.NOMINAL_COLUMNS]

    for idx, column in enumerate(present_nominal_columns):
        present_nominal_columns_idx_dict[column] = idx + len(remaining_columns) 

    with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NOMINAL_COLUMNS_IDX_FILENAME),"wb") as file_handle:
        pickle.dump(present_nominal_columns_idx_dict,file_handle)



def save_ordinal_columns_idx():

    present_ordinal_columns_idx_dict = dict()
    all_relevant_columns = config.REFINED_COLUMNS.copy()

    for column in config.NOMINAL_COLUMNS:

        if column in all_relevant_columns:
            all_relevant_columns.remove(column)

    for idx, column in enumerate(all_relevant_columns):

        if column not in config.NUMERIC_COLUMNS:
            present_ordinal_columns_idx_dict[column] = idx

    with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ORDINAL_COLUMNS_IDX_FILENAME),"wb") as file_handle:
        pickle.dump(present_ordinal_columns_idx_dict,file_handle)



def save_numeric_columns_idx():

    present_numeric_columns_idx_dict = dict()
    all_relevant_columns = config.REFINED_COLUMNS.copy()

    for column in config.NOMINAL_COLUMNS:

        if column in all_relevant_columns:
            all_relevant_columns.remove(column)

    for idx, column in enumerate(all_relevant_columns):

        if column in config.NUMERIC_COLUMNS:
            present_numeric_columns_idx_dict[column] = idx

    with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NUMERIC_COLUMNS_IDX_FILENAME),"wb") as file_handle:
        pickle.dump(present_numeric_columns_idx_dict,file_handle)

