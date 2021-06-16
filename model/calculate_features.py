# Slope of Gutenberg-Richter curve
import math
import numpy as np
import pandas as pd
from numpy import longdouble, int8
from scipy.optimize import curve_fit


class CalculateFeatures:

    def __init__(self, df, n):

        self.n = n

        # Features

        """
        Description:
        All the seismic features are calculated using the 50 seismic events before the event of interest Et
        Seismic events are grouped in 
        All following features are computed on every group:
        
        firstT              -:  first date
        lastT               -:  last date
        meanMag             -:  mean magnitude
        a                   -:  y-intercept of Gutenberg-Richter curve
        b                   -:  Slope of Gutenberg-Richter curve
        elapsedT            -:  Time elapsed of the last 'n' seismic events
        rateSqrtEnergy      -:  Rate of square root of energy
        meanTDiff           -:  Mean of the differences of the times of the events
        maxMag              -:  Max magnitude
        seismicRateChange   -:  Seismic rate change proposed by Habermann and Wyss
        last7dMaxMag        -:  Max magnitude in the last 7 days
        next7dMaxMag        -:  Max magnitude in the next 7 days
        next14dMaxMag       -:  Max magnitude in the next 14 days
        
        """
        self._f1_columns = ["firstT",
                            "lastT",
                            "meanMag",
                            "a",
                            "b",
                            "elapsedT",
                            "rateSqrtEnergy",
                            "meanTDiff",
                            "maxMag",
                            ]

        self.features = pd.DataFrame([self._apply_features_agg(w) for w in df.rolling(n, min_periods=n) if len(w.index)==n],
                                     columns=self._f1_columns)

        # Vectorization of Features

        all_dates = np.array(df.Datetime, dtype="datetime64[s]")
        all_magnitudes = np.array(np.array(df.Magnitude))

        """
        self.features["seismicRateChange"] = npext.rolling_apply(
            self.seismic_rate_change,
            2,
            np.array(self.features.elapsedT)
        )

        self.features["last7dMaxMag"] = self._dt_max_magnitude(self.features.firstT, np.timedelta64(-7, 'D'), all_dates,
                                                               all_magnitudes)

        self.features["next7dMaxMag"] = self._dt_max_magnitude(self.features.lastT, np.timedelta64(7, 'D'), all_dates,
                                                               all_magnitudes)
        """

    def _apply_features_agg(self, df_group):
        datetime_array = np.array(df_group.Datetime, dtype='datetime64[s]')
        magnitudes_array = np.array(df_group.Magnitude)

        T = self.elapsed_time(datetime_array)
        if len(df_group.index) >= self.n:
            a, b = self._gutenberg_richter_curve_fit(magnitudes_array)
        else:
            a = b = np.NaN
        meanMag = np.mean(magnitudes_array)
        dE = self.rate_square_root_energy(magnitudes_array, T)
        u = self.mean_time_difference(datetime_array)
        max_mag = np.max(magnitudes_array)

        return [
            datetime_array[0],
            datetime_array[-1],
            meanMag,
            a,
            b,
            T,
            dE,
            u,
            max_mag,
        ]

    def get_n(self, magnitude):
        return self.gutenberg_richter_law(magnitude, self.a, self.b)

    # FEATURE DEFINITION

    #
    # a - y-intercept of Gutenberg-Richter curve
    # b - Seismic rate change proposed by Matthews and Reasenberg
    #

    @staticmethod
    def gutenberg_richter_law(m, a, b):
        with np.errstate(over='ignore'):
            return np.power(10, a - b * m, dtype=longdouble)

    def _gutenberg_richter_curve_fit(self, magnitude):
        unique, counts = np.unique(magnitude, return_counts=True)
        view = np.flip(counts, 0)

        if self.n <= 127:
            np.cumsum(view, 0, dtype=int8, out=view)
        else:
            np.cumsum(view, 0, out=view)

        return curve_fit(CalculateFeatures.gutenberg_richter_law, unique, counts)[0]

    @staticmethod
    def gutenberg_richter_curve_fit(magnitude):
        unique, counts = np.unique(magnitude, return_counts=True)
        view = np.flip(counts, 0)
        np.cumsum(view, 0, out=view)

        return curve_fit(CalculateFeatures.gutenberg_richter_law, unique, counts)[0]

    #
    # T - Time elapsed for last “n” seismic events
    #

    @staticmethod
    def elapsed_time(datetimes_array):
        return (datetimes_array[-1] - datetimes_array[0]).astype(int)

    #
    # dE - Rate of Square Root of Energy
    # ( 5.9 + .75 * m ) == sqrt( 11.8 + 1.5 * m )
    #

    @staticmethod
    def rate_square_root_energy(magnitude, t):
        return np.sum(np.power(10, 5.9 + .75 * magnitude)) / t

    #
    # U - Mean time between characteristic events
    #

    @staticmethod
    def mean_time_difference(datetimes):
        return np.mean(np.diff(datetimes).astype(int))

    #
    # Z - # Seismic rate change proposed by Habermann and Wyss
    #

    def seismic_rate_change(self, T):
        return math.sqrt(self.n * abs(math.pow(T[0], 2) - math.pow(T[1], 2)) / (T[0] + T[1]))

    #
    # x6 ( DT_max_m )- # Maximum magnitude earthquake recorded between (Te-dT , Te)
    #

    @staticmethod
    def _dt_max_magnitude(eT_dates, dT, all_dates, all_magnitudes):

        def _positive_dt_filter(_eT):
            return ((_eT + dT) >= all_dates) & (all_dates > _eT)

        def _negative_dt_filter(_eT):
            return ((_eT + dT) <= all_dates) & (all_dates < _eT)

        dt_filter = _positive_dt_filter
        max_mag_dT = np.array([], dtype="float")
        if dT < 0:  # negative
            max_mag_dT = np.append(max_mag_dT, [np.NaN])
            dt_filter = _negative_dt_filter
            eT_dates = eT_dates[1:]
        else:
            eT_dates = eT_dates[:-1]

        for eT in eT_dates:
            eT = np.datetime64(eT, "s")
            max_mag_dT = np.append(max_mag_dT, np.max(all_magnitudes[dt_filter(eT)]))

        if dT > 0:  # positive:
            max_mag_dT = np.append(max_mag_dT, [np.NaN])

        return max_mag_dT
