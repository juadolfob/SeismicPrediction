# Slope of Gutenberg-Richter curve
import math

import numpy as np
import numpy_ext as npext
import pandas as pd
from numpy import longdouble, int8
from scipy.optimize import curve_fit


class CalculateFeatures:

    def __init__(self, df, n):

        # self variables
        self.n = n
        all_dates = np.array(df.Datetime, dtype="datetime64[s]")
        all_magnitudes = np.array(df.Magnitude)
        self.total_events = len(df.index)
        self.a, self.b = self.gutenberg_richter_curve_fit(all_magnitudes)



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
        zSeismicRateChange  -:  Seismic rate change proposed by Habermann and Wyss
        bSeismicRateChange  -:  Seismic rate change proposed by Matthews and Reasenberg
        last7dMaxMag        -:  Max magnitude in the last 7 days
        next7dMaxMag        -:  Max magnitude in the next 7 days
        next14dMaxMag       -:  Max magnitude in the next 14 days
        
        """
        self._f1_columns = ["old_index",
                            "firstT",
                            "lastT",
                            "meanMag",
                            "a",
                            "b",
                            "elapsedT",
                            "rateSqrtEnergy",
                            "meanTDiff",
                            "maxMag",
                            ]

        self.features = pd.DataFrame(
            [self._apply_features_agg(w) for w in df.rolling(n, min_periods=n) if len(w.index) == n],
            columns=self._f1_columns)

        # Vectorization of Features

        def _rolling_2_features_apply(_rolling_2_index):
            z_seismic_rate_change = [np.NaN]
            b_seismic_rate_change = [np.NaN]
            for index in _rolling_2_indexes:
                array_elapsed_T = np.array(self.features.elapsedT.iloc[index])
                z_seismic_rate_change.append(self.z_seismic_rate_change(array_elapsed_T))
                b_seismic_rate_change.append(self.b_seismic_rate_change(array_elapsed_T))
            return (z_seismic_rate_change,
                    b_seismic_rate_change)

        _rolling_2_indexes = npext.rolling(np.array(self.features.index.to_numpy()), 2, as_array=True, skip_na=True)

        self.features[["zSeismicRateChange",
                       "bSeismicRateChange"]] = np.array(_rolling_2_features_apply(_rolling_2_indexes)).T

        self.features["last7dMaxMag"] = self._dt_max_magnitude(self.features.lastT, np.timedelta64(-7, 'D'), all_dates,
                                                               all_magnitudes)

        self.features["next14dMaxMag"] = self._dt_max_magnitude(self.features.lastT, np.timedelta64(14, 'D'), all_dates,
                                                                all_magnitudes)
        # todo cut fron and back row < 7 days from limits
        self.features.drop([0, len(self.features.index) - 1], axis=0, inplace=True)

    def _apply_features_agg(self, df_group):
        old_index = df_group.index.stop - 1
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
            old_index,
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
        """ to print
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        a, b = curve_fit(CalculateFeatures.gutenberg_richter_law, unique, counts)[0]

        x = unique
        y = counts
        plt.figure()
        plt.plot(x, y, 'ko', label="Original Data")
        plt.plot(x, CalculateFeatures.gutenberg_richter_law(x, a, b), 'r-', label="Fitted Curve")
        plt.legend()
        plt.show()
        """
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

    def z_seismic_rate_change(self, T):
        return math.sqrt(self.n * abs(math.pow(T[0], 2) - math.pow(T[1], 2)) / (T[0] + T[1]))

    #
    # Z - # Seismic rate change proposed by Habermann and Wyss
    #

    def b_seismic_rate_change(self, T):
        return (self.n * (T[1] - T[0])) / math.sqrt(self.n * T[0] + T[1])

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
