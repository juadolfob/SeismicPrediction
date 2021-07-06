"""
function to facilitate loading of data
"""
import model
import pandas as pd
import numpy as np


def load_data(file):
    """
    loads data and returns pandas dataframe
    :return: pandas dataframe
    """
    df = pd.read_csv(file, delimiter=',', dtype=model.DATA_DTYPES, parse_dates=model.DATA_DATETIME,
                     date_parser=np.datetime64)
    df.reset_index(drop=True)
    return df
