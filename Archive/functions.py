# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


"""
    Bu dosyada makine öğrenmesi için yapılacak memory optimizasyonu için bulunan fonksiyonlar yer almaktadır.
"""


def memory_management(df:DataFrame, **kwargs) -> DataFrame:
    if kwargs.get('categorical_columns') is not None:
        categorical_columns = df[kwargs['categorical_columns']].columns
        for column in categorical_columns:
            df[column] = df[column].astype('category')

    if kwargs.get('continuous_columns') is not None:
        continuous_columns = df[kwargs['continuous_columns']].columns
        for column in continuous_columns:
            if ('int' in str(df[column].dtype)):
                for dtype in [np.int8, np.int16, np.int32]:
                    if max(df[column]) < np.iinfo(dtype).max:
                        df[column] = df[column].astype(dtype)
                        break
            else:
                df[column] = df[column].astype('float32')

    df['churn'] = df['churn'].astype('uint8')
    return df


def bool_to_binary(df:DataFrame, **kwargs)->DataFrame:
    continuous_columns = df[kwargs['continuous_columns']].columns
    for column in continuous_columns:
        df[column] = df[column].astype(np.int8)
    return df

def pred_to_binary(pred_val)->list:
    return [1 if val > -1.019 else 0 for val in list(pred_val)]


