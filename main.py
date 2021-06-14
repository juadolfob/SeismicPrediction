import math
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/DATA_2.csv', delimiter=',', parse_dates=['Datetime'])

from model import indicator


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

GRC = indicator.gutenberg_richter_curve(df.Magnitude)
x = GRC.mapped_mag[:,0]
y = GRC.mapped_mag[:,1]
popt=(GRC.a,GRC.b)
plt.figure()
plt.plot(x, y, 'ko', label="Original Noised Data")
plt.plot(x, GRC.gutenberg_richter_law(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

popt
