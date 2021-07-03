import numpy as np


def feature_selection(df, threshold):
    columns = df.columns.to_list()
    corr_features_above_threshold = set()
    corr_features_below_threshold = set(columns)
    corr_matrix_masked = np.tril(np.corrcoef(df.to_numpy().T), -1)
    for i in range(len(corr_matrix_masked)):
        for j in range(i):
            if abs(corr_matrix_masked[i, j]) > threshold:
                corr_features_above_threshold.add(columns[i])
                corr_features_below_threshold.discard(columns[i])
    return tuple(map(list, (corr_features_above_threshold, corr_features_below_threshold)))
