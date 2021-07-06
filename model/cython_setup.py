from distutils.core import setup
from Cython.Build import cythonize

setup(name="model", ext_modules=cythonize(['cython_loop.pyx'], language_level="3"), )
