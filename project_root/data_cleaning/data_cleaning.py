from config import config
import numpy as np



def drop_rows_columns(data):

    missing_vals_df = data.isna().sum()

    if len(missing_vals_df[missing_vals_df == data.shape[0]].index) > 0:
        data.drop(labels=list(missing_vals_df[missing_vals_df == data.shape[0]].index),axis=1,inplace=True)

    if len(missing_vals_df.values == np.array([missing_vals_df.values[0]]*data.shape[1])):
        data.dropna(axis=0,inplace=True)

    data.drop_duplicates(inplace=True,ignore_index=True)

    return data




def clean_columns(data):

    for column in config.COLUMNS_TO_CLEAN:

        data[column].replace(to_replace="?",value=data[column].value_counts().index[0],inplace=True)

        if column == "Ram":
            data[column] = data[column].apply(lambda x: float(x.split("GB")[0]))

        if column == "Weight":
            data[column] = data[column].apply(lambda x: float(x.split("kg")[0]))

        if column == "Inches":
            data[column] = data[column].astype("float64")

    return data
