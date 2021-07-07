import numpy as np
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
C2_COLUMNS = ["meanMag", "maxAMag", "maxEMag", "magDef", "a", "b", "bStd", "grcStd",
              "elapsedT", "rateSqrtEnergy", "meanT", "meanTStd", "pMag"]
F1_COLUMNS = list(chain(C1_COLUMNS, C2_COLUMNS, C0_COLUMNS))
F2_COLUMNS = ["zSeismicRateChange", "bSeismicRateChange"]
F3_COLUMNS = ["lastDMaxMag"]
F4_COLUMNS = ["nextDMaxMag"]
F5_COLUMNS = ["gTMag"]

# FEATURES

FEATURES_DTYPES = {feature: int for feature in C0_COLUMNS} | \
                  {feature: np.float for feature in list(chain(C2_COLUMNS, F2_COLUMNS, F3_COLUMNS, F4_COLUMNS))} | \
                  {feature: np.bool for feature in F5_COLUMNS}

FEATURES_DATETIME = C1_COLUMNS
FEATURES = F1_COLUMNS + F2_COLUMNS + F3_COLUMNS

TARGETS_CONTINUOUS = F4_COLUMNS
TARGETS_CATEGORICAL = F5_COLUMNS

ALL_FEATURES = FEATURES + TARGETS_CATEGORICAL + TARGETS_CATEGORICAL
