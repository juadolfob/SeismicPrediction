from . import parameters
from .calculate_features import *
from .feature_selection import *

FEATURES = ['meanMag', 'maxAMag', 'maxEMag', 'magDef', 'a', 'b', 'bStd', 'grcStd', 'elapsedT', 'rateSqrtEnergy',
            'meanT', 'meanTStd', 'pTMag', 'zSeismicRateChange', 'bSeismicRateChange', 'lastDMaxMag']

TARGETS = ['nextDMaxMag']
TARGETS = ['nextDGTMag']

ALL_FEATURES = FEATURES + TARGETS
