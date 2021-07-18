"""
Feature Selection

Feature selection is one of the main challenges in analyzing high-throughput genomic data. Minimum redundancy maximum
relevance (mRMR) is a particularly fast feature selection method for finding a set of both relevant and complementary
features. The Pymrmre package, extend the mRMR technique by using an ensemble approach to better explore the feature
space and build more robust predictors. To deal with the computational complexity of the ensemble approach, the main
functions of the package are implemented and parallelized in C++ using openMP Application Programming Interface. The
package also supports making best selections with some fixed-selected features.

This program and the respective minimum Redundancy Maximum Relevance (mRMR)
    algorithm were developed by Hanchuan Peng <hanchuan.peng@gmail.com>for
     the paper
     "Feature selection based on mutual information: criteria of
      max-dependency, max-relevance, and min-redundancy,"
      Hanchuan Peng, Fuhui Long, and Chris Ding,
      IEEE Transactions on Pattern Analysis and Machine Intelligence,
      Vol. 27, No. 8, pp.1226-1238, 2005.
"""
import numpy as np
import pandas as pd
import pymrmre


class SelectFeatures:
    """
    Feature Selection
    """

    def __init__(self,
                 features: pd.DataFrame,
                 targets: pd.DataFrame,
                 corr_threshold: float,
                 solution_length: int = 1,
                 solution_count: int = 1,
                 category_features: list = [],
                 fixed_features: list = [],
                 method: str = 'exhaustive',
                 estimator: str = 'pearson',
                 return_index: bool = False,
                 return_with_fixed: bool = True):
        """
        :param features: Pandas dataframe, the input dataset
        :param targets: Pandas dataframe, the target features
        :param corr_threshold: Threshold for calculation of solution_length, overrides any solution_length value
        :param fixed_features: List, the list of fixed features (column names)
        :param category_features: List, the list of features of categorical type (column names)
        :param solution_length: Integer, the number of features contained in one solution
        :param solution_count: Integer, the number of solutions to be returned
        :param method: String, the different ways to run the algorithm, exhaustive or bootstrap
        :param estimator: String, the way of computing continuous estimators
        :param return_index: Boolean, to determine whether the solution contains the indices or column names of selected
         features
        :param return_with_fixed: Boolean, to determine whether the solution contains the fixed selected features
        """
        if corr_threshold:
            solution_length = self.calculate_solution_length(features, corr_threshold)
        self.selected_features = pymrmre.mrmr_ensemble(features=features,
                                                       targets=targets,
                                                       solution_length=solution_length,
                                                       solution_count=solution_count,
                                                       category_features=category_features,
                                                       fixed_features=fixed_features,
                                                       method=method,
                                                       estimator=estimator,
                                                       return_index=return_index,
                                                       return_with_fixed=return_with_fixed)[0]

    @staticmethod
    def calculate_solution_length(array: np.array, corr_threshold: float):
        """
        Uses correlation between features and a threshold to estimate how many features should be kept
        :return: estimated_solution_length
        """
        return array.shape[1] - np.sum(np.tril(np.corrcoef(array.T), -1) >= corr_threshold)

    def __repr__(self):
        return "SelectedFeatures(%s)" % self.selected_features.__str__()

    def __str__(self):
        return self.selected_features.__str__()

    def __getitem__(self, item):
        return self.selected_features.__getitem__(item)

    def __iter__(self):
        return self.selected_features.__iter__()

    def __len__(self):
        return self.selected_features.__len__()

