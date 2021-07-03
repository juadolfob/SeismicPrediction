import io

import matplotlib
from IPython.core.display import display
from PIL import Image

matplotlib.use('Qt5Cairo')


def showfig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(io.BytesIO(buf.read()))
    display(img)