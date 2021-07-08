"""
utils for jupyter notebooks
"""

import io
import matplotlib
from IPython.core.display import display
from PIL import Image
from model import load_features_from_cache, load_data, CalculateFeatures, features_in_cache

matplotlib.use('Qt5Cairo')


def show_fig(fig: matplotlib.figure.Figure):
    """
    shows fig from buffer, allows to render multiple times the same figure as an image
    :param fig:
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(io.BytesIO(buf.read()))
    display(img)


def get_features():
    """

    """
    # delete cached features to generate new features
    if features_in_cache():
        features = load_features_from_cache()
    else:
        df = load_data('../../data/DATA_3_south.csv')
        features =  CalculateFeatures(df, 50, trim_features=True, mag_threshold=6).features
        features.to_csv("../../model/features_cache/features.csv", index=False)
    return features


"""
embedding = Isomap(n_jobs=-1)
X_transformed = embedding.fit_transform(X_norm32, Y_CATEGORICAL)

"""

"""
%matplotlib qt

fig, ax = plt.subplots()
num_ticks = np.round((Y_CONTINUOUS_Scaler.data_max_ - Y_CONTINUOUS_Scaler.data_min_) * 10 + 1).astype(int)[0]
labels = np.round(np.linspace(Y_CONTINUOUS_Scaler.data_min_, Y_CONTINUOUS_Scaler.data_max_, num=num_ticks).T[0], 1)
ticks = np.linspace(0, 1, num=num_ticks)[np.isin(labels, np.unique(Y_CONTINUOUS))]
cax = plt.scatter(X_transformed.T[0], X_transformed.T[1], s=sizes, color=colors_continuous)

cbar = fig.colorbar(cax, ticks=ticks)
cbar.ax.set_yticklabels(np.unique(Y_CONTINUOUS))
plt.plot()
"""
