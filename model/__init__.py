"""

"""
from itertools import chain

from tensorflow import keras

from . import parameters
from .calculate_features import *
from .feature_selection import *
from .load_data import *
from .predictor import *
from .predictor import *

# DATA

DATA_DTYPES = {'Magnitude': np.float64, 'Latitude': np.float64, 'Longitude': np.float64, 'Depth': np.float64}
DATA_DATETIME = ['Datetime']

# MODEL


C0_COLUMNS = ["old_index"]
C1_COLUMNS = ["firstT", "lastT"]
C2_COLUMNS = ["elapsedT"]
C3_COLUMNS = ["meanMag", "maxMag", "rateSqrtEnergy", "u", "c", "a_lsq", "a_mlk", "b_lsq",
              "b_mlk", "maxEMag_lsq", "maxEMag_mlk", "magDef_lsq", "mag_def_mlk", "bStd_lsq",
              "bStd_mlk", "grcStd_lsq", "grcStd_mlk", "pMag_lsq", "pMag_mlk"]
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


ALL_FEATURES = FEATURES + TARGETS.CONTINUOUS + TARGETS.CATEGORICAL

# MODELS

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
keras.metrics.Recall(name='r2_score'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
]

BATCH_SIZE = 4096
EPOCHS = 200
