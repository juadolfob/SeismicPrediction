"""
Calculates Features
"""
import math
from collections import Iterable

import numpy as np
import numpy_ext as npext
import pandas as pd
from numpy import longdouble
from scipy.optimize import curve_fit

import model


class CalculateFeatures:
    """
        Description:
        All the seismic features are calculated using the 50 seismic events before the event of interest Et
        Seismic events are grouped in
        All following features are computed on every group:

        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

        firstT              -:  first date
        lastT               -:  last date
        meanMag             -:  M_mean | mean magnitude
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
        lastDMaxMag -:  x6 | Max magnitude in the last n days
        nextDMaxMag -:  Max magnitude in the next n days
        pTMag[mag]          -:  x7 | Probability of an earthquake equal or greater than [mag:float]
        tTMag[mag]          -:  Te | Time-to-event in days of an earthquake equal or greater than [mag:float] # todo

        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

        """

    def __init__(self,
                 df: pd.DataFrame,
                 window_size,
                 mag_threshold,
                 trim_features: bool = False,
                 days_forward=14,
                 days_backward=7,
                 ):

        # self variables
        self.window_size = window_size  # window size
        self.all_dates = np.array(df.Datetime, dtype="datetime64[s]")
        all_magnitudes = np.array(df.Magnitude)
        self.firstT = np.datetime64(df.Datetime.iloc[0], "s")
        self.lastT = np.datetime64(df.Datetime.iloc[-1], "s")
        self.daysForward = np.timedelta64(days_forward, 'D')
        self.daysBackward = np.timedelta64(days_backward, 'D')
        self.mag_threshold = mag_threshold

        # FEATURES

        # F1_COLUMNS
        self.features = pd.DataFrame(
            [self._apply_F1_features(w) for w in df.rolling(window_size, min_periods=window_size) if
             len(w.index) == window_size],
            columns=model.F1_COLUMNS)

        self._rolling_2_indexes = npext.rolling(self.features.index.to_numpy(), 2, as_array=True, skip_na=True)
        
        # F2_COLUMNS
        # todo try apply for last n, not for last window with n events, and compare results
        self.features[model.F2_COLUMNS] = np.array(self._rolling_2_features_apply(self._rolling_2_indexes)).T

        features_last_t_array = np.array(self.features.lastT, dtype='datetime64[s]')
        # F3_COLUMNS
        self.features[model.F3_COLUMNS[0]] = self._dt_max_magnitude(features_last_t_array,
                                                                    -self.daysBackward, self.all_dates,
                                                                    all_magnitudes)
        # F4_COLUMNS
        self.features[model.F4_COLUMNS[0]] = self._dt_max_magnitude(features_last_t_array,
                                                                    self.daysForward,
                                                                    self.all_dates,
                                                                    all_magnitudes)
        # F5_COLUMNS
        self.features[model.F5_COLUMNS[0]] = self.features[model.F4_COLUMNS] >= mag_threshold

        # TRIM TIME AND Nan's
        if trim_features:
            self._trim_features(features_last_t_array)

        self.features.reset_index(inplace=True, drop=True)

    def _apply_F1_features(self, df_group):
        datetime_array = np.array(df_group.Datetime, dtype='datetime64[s]')
        magnitudes_array = np.array(df_group.Magnitude)
        this_window_size = len(df_group.index)
        unique_M, unique_N = self._cumcount_sorted_unique(magnitudes_array)
        unique_n = unique_M.shape[0]
        log_unique_N = np.log10(unique_N.astype(longdouble), dtype=longdouble)

        # F1 features

        oldIndex = df_group.index[-1]
        firstT = datetime_array[0]
        lastT = datetime_array[-1]
        elapsedT = self.elapsed_time(datetime_array)
        meanMag = np.mean(magnitudes_array)
        maxMag = np.max(magnitudes_array)
        rateSqrtEnergy = self.rate_square_root_energy(magnitudes_array, elapsedT)
        u = self.mean_time_difference(datetime_array)
        c = self.mean_t_deviation(datetime_array, u)

        b_lsq = self._b_lsq(log_unique_N, unique_M, unique_n)
        a_lsq = self._a_lsq(log_unique_N, unique_M, unique_n, b_lsq)
        b_mlk = np.log10(np.e) / (meanMag - unique_M[0])
        a_mlk = np.log10(np.sum(unique_N)) + b_mlk * unique_M[0]

        # lsq

        maxEMag_lsq = a_lsq / b_lsq
        magDef_lsq = self.magnitude_deficit(maxMag, a_lsq, b_lsq)
        bStd_lsq = self._b_std(unique_M, meanMag, this_window_size, b_lsq)
        grcStd_lsq = self.mean_square_deviation(unique_M, unique_N, this_window_size, a_lsq, b_lsq)
        pMag_lsq = self.p_magnitude(b_lsq, 6.0)

        # mlk

        maxEMag_mlk = a_mlk / b_mlk
        magDef_mlk = self.magnitude_deficit(maxMag, a_mlk, b_mlk)
        bStd_mlk = self._b_std(unique_M, meanMag, this_window_size, b_mlk)
        grcStd_mlk = self.mean_square_deviation(unique_M, unique_N, this_window_size, a_mlk, b_mlk)
        pMag_mlk = self.p_magnitude(b_mlk, 6.0)

        return [
            oldIndex,
            firstT,
            lastT,
            elapsedT,
            meanMag,
            maxMag,
            rateSqrtEnergy,
            u,
            c,
            a_lsq,
            a_mlk,
            b_lsq,
            b_mlk,
            maxEMag_lsq,
            maxEMag_mlk,
            magDef_lsq,
            magDef_mlk,
            bStd_lsq,
            bStd_mlk,
            grcStd_lsq,
            grcStd_mlk,
            pMag_lsq,
            pMag_mlk,
        ]

    @staticmethod
    def _b_lsq(log_unique_N, unique_M, unique_n):
        return (unique_n * np.sum(unique_M * log_unique_N) - np.sum(unique_M) * np.sum(log_unique_N)) / (
                    np.power(np.sum(unique_M), 2) - (unique_n * np.sum(np.power(unique_M, 2))))

    @staticmethod
    def b_lsq(magnitude_array):
        unique_M, unique_N = model.CalculateFeatures._cumcount_sorted_unique(magnitude_array)
        unique_n = unique_M.shape[0]
        log_unique_N = np.log10(unique_N.astype(longdouble), dtype=longdouble)
        return model.CalculateFeatures._b_lsq(log_unique_N, unique_M, unique_n)

    @staticmethod
    def _a_lsq(log_unique_N, unique_M, unique_n, b_lsq):
        return np.sum(log_unique_N + b_lsq * unique_M) / unique_n

    @staticmethod
    def a_lsq(magnitude_array):
        unique_M, unique_N = model.CalculateFeatures._cumcount_sorted_unique(magnitude_array)
        unique_n = unique_M.shape[0]
        log_unique_N = np.log10(unique_N.astype(longdouble), dtype=longdouble)
        b_lsq = CalculateFeatures._b_lsq(log_unique_N, unique_M, unique_n)
        return CalculateFeatures._a_lsq(log_unique_N, unique_M, unique_n, b_lsq)

    def _rolling_2_features_apply(self, _rolling_2_index):
        z_seismic_rate_change = [np.NaN]
        b_seismic_rate_change = [np.NaN]
        for index in self._rolling_2_indexes:
            array_elapsed_t = np.array(self.features.elapsedT.iloc[index])
            z_seismic_rate_change.append(self.z_seismic_rate_change(array_elapsed_t))
            b_seismic_rate_change.append(self.b_seismic_rate_change(array_elapsed_t))
        return (z_seismic_rate_change,
                b_seismic_rate_change)

    def trimmed_features(self):
        features_last_t_array = np.array(self.features.lastT, dtype='datetime64[s]')
        return self._trim_features(features_last_t_array, inplace=False)

    def _trim_features(self, features_last_t_array, inplace=True):
        if inplace:
            self.features.drop([*_drop_index], inplace=True)
            self.features.dropna(inplace=True)
        else:
            _trimmed_features = self.features.drop([*_drop_index])
            return _trimmed_features.dropna()

    def _filter_index_dt(self, features_last_t_array):
        return self.features[
            ~  ((features_last_t_array > (self.firstT + self.daysBackward))
                &
                (features_last_t_array < (self.lastT - self.daysForward)))].index

    def get_n(self, magnitude):
        """

        :param magnitude:
        :return:
        """
        return self.gutenberg_richter_law(magnitude, self.a, self.b)

    # FEATURE DEFINITION

    #
    # a - y-intercept of Gutenberg-Richter curve
    # b - Seismic rate change proposed by Matthews and Reasenberg
    #

    @staticmethod
    def gutenberg_richter_law(m, a, b):
        return np.power(10, a - np.multiply(b, m, dtype=longdouble), dtype=longdouble)

    @staticmethod
    def log_gutenberg_richter_law(m, a, b):
        return a - np.multiply(b, m, dtype=longdouble)

    @staticmethod
    def _cumcount_sorted_unique(array: Iterable):
        unique, counts = np.unique(array, return_counts=True)
        view = np.flip(counts, 0)
        np.cumsum(view, 0, out=view)
        return unique, counts

    @staticmethod
    def _gutenberg_richter_curve_fit(unique, counts):
        log_counts = np.log10(counts.astype(longdouble), dtype=longdouble)
        return curve_fit(CalculateFeatures.log_gutenberg_richter_law, unique, log_counts)[0]

    @staticmethod
    def gutenberg_richter_curve_fit(magnitude):
        """
        Use non-linear least squares to fit a gutenberg_richter_curve's values a and b, to data.
        :param magnitude:
        :param n:
        :return:
        """

        unique, counts = CalculateFeatures._cumcount_sorted_unique(magnitude)
        log_counts = np.log10(counts.astype(longdouble), dtype=longdouble)
        return curve_fit(CalculateFeatures.log_gutenberg_richter_law, unique, log_counts)[0]

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
        return math.sqrt(np.sum(np.power(datetimes.astype(float) - u, 2)) / len(datetimes) - 1)

    #
    # z - Seismic rate change proposed by Habermann and Wyss
    #

    def z_seismic_rate_change(self, T):
        return math.sqrt(self.window_size * abs(math.pow(T[0], 2) - math.pow(T[1], 2)) / (T[0] + T[1]))

    #
    # Z - Seismic rate change proposed by Habermann and Wyss
    #

    def b_seismic_rate_change(self, T):
        return (self.window_size * (T[1] - T[0])) / math.sqrt(self.window_size * T[0] + T[1])

    #
    # x6 ( DT_max_m ) - Maximum magnitude earthquake recorded between (Te-dT , Te)
    #

    @staticmethod
    def _dt_max_magnitude(e_t_dates, d_t, all_dates, all_magnitudes):
        def _positive_dt_filter(_e_t):
            return ((_e_t + d_t) >= all_dates) & (all_dates > _e_t)

        def _negative_dt_filter(_e_t):
            return ((_e_t + d_t) <= all_dates) & (all_dates < _e_t)

        dt_filter = _positive_dt_filter
        max_mag_d_t = np.array([], dtype="float")
        if d_t < 0:  # negative
            dt_filter = _negative_dt_filter

        for e_t in e_t_dates:
            e_t = np.datetime64(e_t, "s")
            filtered_mags = all_magnitudes[dt_filter(e_t)]
            if filtered_mags.size > 0:
                max_mag_d_t = np.append(max_mag_d_t, np.max(filtered_mags))
            else:
                max_mag_d_t = np.append(max_mag_d_t, np.NaN)

        return max_mag_d_t

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
    def mean_square_deviation(m_unique, m_count, n_events, a, b):
        return np.sum(np.power(np.log(m_count) - a - b * m_unique, 2)) / n_events - 1

    #
    #  σb ( b_standard_deviation , b_std ) - Standard deviation of b-value
    #

    @staticmethod
    def _b_std(m_unique, m_mean, n_events, b):
        return 2.3 * math.pow(b, 2) * math.sqrt(
            np.sum(np.power(m_unique - m_mean, 2)) / (n_events * (n_events - 1)))

    #
    #  x7 ( p_magnitude , pMag ) - Standard deviation of b-value
    #

    @staticmethod
    def p_magnitude(b, m):
        return pow(10, -b * m)
