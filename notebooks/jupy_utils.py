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
