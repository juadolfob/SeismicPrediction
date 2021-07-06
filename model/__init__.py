from . import parameters
from .calculate_features import *
from .feature_selection import *
from .load_data import *

# DATA
DATA_DTYPES = {'Magnitude': np.float64, 'Latitude': np.float64, 'Longitude': np.float64, 'Depth': np.float64}
DATA_DATETIME = ['Datetime']

# MODEL
F0_COLUMNS = ["old_index", "firstT", "lastT"]
F1_COLUMNS = ["meanMag", "maxAMag", "maxEMag", "magDef", "a", "b", "bStd", "grcStd",
              "elapsedT", "rateSqrtEnergy", "meanT", "meanTStd", "pMag"]
C0_COLUMNS = F0_COLUMNS + F1_COLUMNS
F2_COLUMNS = ["zSeismicRateChange", "bSeismicRateChange"]
F3_COLUMNS = ["lastDMaxMag"]
F4_COLUMNS = ["nextDMaxMag"]
F5_COLUMNS = ["nextDMaxMag"]

# FEATURES
FEATURES = F1_COLUMNS + F2_COLUMNS + F3_COLUMNS

TARGETS_CONTINUOUS = F4_COLUMNS
TARGETS_CATEGORICAL = F5_COLUMNS

ALL_FEATURES = FEATURES + TARGETS_CATEGORICAL + TARGETS_CATEGORICAL
