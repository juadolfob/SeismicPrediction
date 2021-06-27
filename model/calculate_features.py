# Slope of Gutenberg-Richter curve
import math
from collections import Iterable

import numpy as np
import numpy_ext as npext
import pandas as pd
from numpy import longdouble, int8
from scipy.optimize import curve_fit


# noinspection SpellCheckingInspection
class CalculateFeatures:

    def __init__(self, df, n, trimFeatures=False, daysForward=14, daysBackward=7):

        # self variables
        self.n = n
        all_dates = np.array(df.Datetime, dtype="datetime64[s]")
        all_magnitudes = np.array(df.Magnitude)
        self.total_events = len(df.index)
        self.a, self.b = self.gutenberg_richter_curve_fit(all_magnitudes)
        self.firstT = np.datetime64(df.Datetime.iloc[0], "s")
        self.lastT = np.datetime64(df.Datetime.iloc[-1], "s")
        self.daysForward = np.timedelta64(daysForward, 'D')
        self.daysBackward = np.timedelta64(daysBackward, 'D')
        # Features

        """
        Description:
        All the seismic features are calculated using the 50 seismic events before the event of interest Et
        Seismic events are grouped in 
        All following features are computed on every group:
        
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        
        firstT              -:  first date
        lastT               -:  last date
        meanMag             -:  Mmean | mean magnitude
        a                   -:  a | y-intercept of Gutenberg-Richter curve
        b                   -:  b | Slope of Gutenberg-Richter curve
        bStd                -:  σb | Standard deviation of b value
        grcStd              -:  η | Mean square deviation
        elapsedT            -:  T | Time elapsed of the last 'n' seismic events        
        rateSqrtEnergy      -:  dE1/2 | Rate of square root of energy
        meanT               -:  μ | Mean of the differences of the times of each event
        meanTStd            -:  c | Standard Deviation of the differences of the times of each event
        maxAMag             -:  Max actual magnitude
        maxEMag             -:  Max expected magnitude
        magDef              -:  ΔM | Magnitude deficit
        zSeismicRateChange  -:  z | Seismic rate change proposed by Habermann and Wyss
        bSeismicRateChange  -:  β | Seismic rate change proposed by Matthews and Reasenberg
        last[backDT]MaxMag  -:  x6 | Max magnitude in the last [backDT] days
        next[frontDT]MaxMag -:  Maxmagnitude in the next [frontDT] days
        pMag[mag]           -:  x7 | Probability of an earthquaky euqual or greater than [mag]
        
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        
        """

        self._f1_columns = ["old_index",
                            "firstT",
                            "lastT",
                            "meanMag",
                            "maxAMag",
                            "maxEMag",
                            "magDef",
                            "a",
                            "b",
                            "bStd",
                            "grcStd",
                            "elapsedT",
                            "rateSqrtEnergy",
                            "meanT",
                            "meanTStd",
                            "pMag6",
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

        _rolling_2_indexes = npext.rolling(self.features.index.to_numpy(), 2, as_array=True, skip_na=True)

        self.features[["zSeismicRateChange",
                       "bSeismicRateChange"]] = np.array(_rolling_2_features_apply(_rolling_2_indexes)).T

        # DELTA TIME FEATURES

        features_lastT_array = np.array(self.features.lastT, dtype='datetime64[s]')

        self.features["last" + str(daysBackward) + "dMaxMag"] = self._dt_max_magnitude(features_lastT_array,
                                                                                       -self.daysBackward, all_dates,
                                                                                       all_magnitudes)

        self.features["next" + str(daysForward) + "dMaxMag"] = self._dt_max_magnitude(features_lastT_array,
                                                                                      self.daysForward,
                                                                                      all_dates,
                                                                                      all_magnitudes)
        if trimFeatures:
            self._trim_features(features_lastT_array)

        self.features.reset_index(inplace=True, drop=True)

    def _apply_features_agg(self, df_group):
        old_index = df_group.index.stop - 1
        datetime_array = np.array(df_group.Datetime, dtype='datetime64[s]')
        magnitudes_array = np.array(df_group.Magnitude)
        T = self.elapsed_time(datetime_array)
        n_events = len(df_group.index)
        meanMag = np.mean(magnitudes_array)

        if n_events >= self.n:
            unique, count = self._cumcount_sorted_unique(magnitudes_array)
            a, b = self._gutenberg_richter_curve_fit(unique, count)
            grc_std = self.mean_square_deviation(unique, count, n_events, a, b)
            b_std = self._b_std(unique, meanMag, n_events, b)
        else:
            a = np.NaN
            b = np.NaN
            grc_std = np.NaN
            b_std = np.NaN

        max_actual_mag = np.max(magnitudes_array)
        max_expected_mag = a / b
        dE = self.rate_square_root_energy(magnitudes_array, T)
        u = self.mean_time_difference(datetime_array)
        mag_def = self.magnitude_deficit(max_actual_mag, a, b)
        p_mag_6 = self.p_magnitude(b, 6.0)
        c = self.mean_t_deviation(datetime_array, u)
        return [
            old_index,
            datetime_array[0],
            datetime_array[-1],
            meanMag,
            max_actual_mag,
            max_expected_mag,
            mag_def,
            a,
            b,
            b_std,
            grc_std,
            T,
            dE,
            u,
            c,
            p_mag_6,
        ]

    def get_trim_features(self):
        features_lastT_array = np.array(self.features.lastT, dtype='datetime64[s]')
        _dropindex = self._filter_index_dt(features_lastT_array)
        return self.features.drop([0, len(self.features.index) - 1, *_dropindex], axis=0)

    def trim_features(self):
        features_lastT_array = np.array(self.features.lastT, dtype='datetime64[s]')
        _dropindex = self._filter_index_dt(features_lastT_array)
        self.features.drop([0, len(self.features.index) - 1, *_dropindex], axis=0, inplace=True)

    def _trim_features(self, features_lastT_array):
        _dropindex = self._filter_index_dt(features_lastT_array)
        self.features.drop([0, len(self.features.index) - 1, *_dropindex], axis=0, inplace=True)

    def _filter_index_dt(self, features_lastT_array):
        return self.features[
            ~  ((features_lastT_array > (self.firstT + self.daysBackward))
                &
                (features_lastT_array < (self.lastT - self.daysForward)))].index

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

    @staticmethod
    def _cumcount_sorted_unique(array: Iterable, n: int = None):
        unique, counts = np.unique(array, return_counts=True)
        view = np.flip(counts, 0)
        if n:
            if n <= 127:
                np.cumsum(view, 0, dtype=int8, out=view)
            else:
                np.cumsum(view, 0, out=view)
        np.cumsum(view, 0, out=view)
        return unique, counts

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

    @staticmethod
    def _gutenberg_richter_curve_fit(unique, counts):
        return curve_fit(CalculateFeatures.gutenberg_richter_law, unique, counts)[0]

    @staticmethod
    def gutenberg_richter_curve_fit(magnitude, n=None):
        unique, counts = CalculateFeatures._cumcount_sorted_unique(magnitude, n)

        return curve_fit(CalculateFeatures.gutenberg_richter_law, unique, counts)[0]

    #
    # T - Time elapsed for last “n” seismic events
    #

    @staticmethod
    def elapsed_time(datetimes_array):
        return (datetimes_array[-1] - datetimes_array[0]).astype(int)

    #
    # dE1/2 - Rate of Square Root of Energy
    # ( 5.9 + .75 * m ) == sqrt( 11.8 + 1.5 * m )
    #

    @staticmethod
    def rate_square_root_energy(magnitude, t):
        return np.sum(np.power(10, 5.9 + .75 * magnitude)) / t

    #
    # μ - Mean time between characteristic events
    #

    @staticmethod
    def mean_time_difference(datetimes):
        return np.mean(np.diff(datetimes).astype(float))

    #
    # c - Deviation from mean time
    #

    @staticmethod
    def mean_t_deviation(datetimes, u):
        return math.sqrt(np.sum(np.power(datetimes.astype(float) - u, 2))/len(datetimes)-1)

    #
    # z - Seismic rate change proposed by Habermann and Wyss
    #

    def z_seismic_rate_change(self, T):
        return math.sqrt(self.n * abs(math.pow(T[0], 2) - math.pow(T[1], 2)) / (T[0] + T[1]))

    #
    # Z - Seismic rate change proposed by Habermann and Wyss
    #

    def b_seismic_rate_change(self, T):
        return (self.n * (T[1] - T[0])) / math.sqrt(self.n * T[0] + T[1])

    #
    # x6 ( DT_max_m ) - Maximum magnitude earthquake recorded between (Te-dT , Te)
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
            dt_filter = _negative_dt_filter

        for eT in eT_dates:
            eT = np.datetime64(eT, "s")
            filtered_mags = all_magnitudes[dt_filter(eT)]
            if filtered_mags.size > 0:
                max_mag_dT = np.append(max_mag_dT, np.max(filtered_mags))
            else:
                max_mag_dT = np.append(max_mag_dT, np.NaN)

        return max_mag_dT

    #
    #  ΔM ( magnitude_deficit ) - difference between the maximum actual magnitude and maximum expected magnitude
    #

    @staticmethod
    def magnitude_deficit(max_mag, a, b):
        return max_mag - a / b

    #
    #  η ( mean_square_deviation ) - difference between the maximum actual magnitude and maximum expected magnitude
    #

    @staticmethod
    def mean_square_deviation(M, N, n, a, b):
        return np.sum(np.power(np.log(N) - a - b * M, 2)) / n - 1

    #
    #  σb ( b_standard_deviation , b_std ) - Standard deviation of b-value
    #

    @staticmethod
    def _b_std(M, M_mean, n, b):
        return 2.3 * math.pow(b, 2) * math.sqrt(np.sum(np.power(M - M_mean, 2)) / n * (n - 1))

    #
    #  x7 ( p_magnitude , pMag ) - Standard deviation of b-value
    #

    @staticmethod
    def p_magnitude(b, m):
        return pow(10, -b * m)
