# Slope of Gutenberg-Richter curve

import numpy as np
from collections import Counter

from numpy import longdouble
from scipy.optimize import curve_fit


# b - Seismic rate change proposed by Matthews and Reasenberg
# a - y-intercept of Gutenberg-Richter curve


class Features:

    def __init__(self, magnitude):
        mapped_mag = np.array(list(dict(Counter(np.array(magnitude))).items()))
        mapped_mag = mapped_mag[np.argsort(mapped_mag[:, 0])]
        mapped_mag[:, 1] = mapped_mag[::-1, 1].cumsum()[::-1]
        self.mapped_mag = mapped_mag
        self.a, self.b = self.gutenberg_richter_curve_fit()

    @staticmethod
    def gutenberg_richter_law(m, a, b):
        with np.errstate(over='ignore'):
            return np.power(10, a - b * m, dtype=longdouble)

    def gutenberg_richter_curve_fit(self):
        x = self.mapped_mag[:, 0]
        y = self.mapped_mag[:, 1]
        popt, _ = curve_fit(Features.gutenberg_richter_law, x, y)
        return popt

    def get_n(self, magnitude):
        return self.gutenberg_richter_law(magnitude, self.a, self.b)


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

GRC = features.gutenberg_richter_curve(df.Magnitude)
x=GRC.mapped_mag[:,0]
y=GRC.mapped_mag[:,1]
plt.figure()
plt.plot(x, y, 'ko', label="Original Data")
plt.plot(x, GRC.gutenberg_richter_law(x,GRC.a,GRC.b), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

"""
