"""

"""
from itertools import chain

from . import parameters
from .calculate_features import *
from .feature_selection import *
from .load_data import *

# DATA

DATA_DTYPES = {'Magnitude': np.float64, 'Latitude': np.float64, 'Longitude': np.float64, 'Depth': np.float64}
DATA_DATETIME = ['Datetime']

# MODEL


C0_COLUMNS = ["old_index"]
C1_COLUMNS = ["firstT", "lastT"]
C2_COLUMNS = ["elapsedT"]
C3_COLUMNS = ["meanMag", "maxAMag", "maxEMag", "magDef", "a", "b", "bStd", "grcStd",
              "rateSqrtEnergy", "meanT", "meanTStd", "pMag"]
F1_COLUMNS = list(chain(C0_COLUMNS, C1_COLUMNS, C2_COLUMNS, C3_COLUMNS))
F2_COLUMNS = ["zSeismicRateChange", "bSeismicRateChange"]
F3_COLUMNS = ["lastDMaxMag"]
F4_COLUMNS = ["nextDMaxMag"]
F5_COLUMNS = ["nextDMaxMagT"]

MODEL_DTYPES = {feature: int for feature in C0_COLUMNS + C2_COLUMNS} | \
               {feature: np.float for feature in list(chain(C3_COLUMNS, F2_COLUMNS, F3_COLUMNS, F4_COLUMNS))} | \
               {feature: np.bool for feature in F5_COLUMNS}
MODEL_DATETIME = C1_COLUMNS

# FEATURES


FEATURES = list(chain(C2_COLUMNS, C3_COLUMNS, F2_COLUMNS, F3_COLUMNS))


class TARGETS:
    """
    TARGETS
    """
    CONTINUOUS = F4_COLUMNS
    CATEGORICAL = F5_COLUMNS
    ALL = F4_COLUMNS + F5_COLUMNS


ALL_FEATURES = FEATURES + TARGETS.CATEGORICAL + TARGETS.CATEGORICAL
