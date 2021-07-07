"""
function to facilitate loading of data
"""
import os
import pathlib


import pandas as pd
import numpy as np

import model

_file_parent_path = pathlib.Path(__file__).parent.resolve()
cached_features_path = _file_parent_path.joinpath('features_cache/features.csv')

def load_data(file):
    """
    loads data and returns pandas dataframe
    :return: pandas dataframe
    """
    df = pd.read_csv(file, delimiter=',', dtype=model.DATA_DTYPES, parse_dates=model.DATA_DATETIME,
                     date_parser=np.datetime64)
    df.reset_index(drop=True)
    return df


def features_in_cache():
    """

    :return:
    """
    return os.path.isfile(cached_features_path)

def load_features_from_cache():
    """
    loads data and returns pandas dataframe
    :return: pandas dataframe
    """

    if features_in_cache():
        df = pd.read_csv(cached_features_path, delimiter=',', dtype=model.DATA_DTYPES,
                         parse_dates=model.DATA_DATETIME, date_parser=np.datetime64)
    else:
        return False

    return df.reset_index(drop=True)

