import itertools
from time import perf_counter

import numpy as np
import numpy_ext as npext
import pandas as pd
import pyximport


pyximport.install(language_level=3,
                  setup_args={'include_dirs': [np.get_include(), npext.np.get_include()]})

