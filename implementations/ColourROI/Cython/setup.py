from distutils.core import setup
from Cython.Build import cythonize
import numpy

print(numpy.get_include())

setup(
  name = 'ROI Find Colour',
  ext_modules = cythonize("ROIFindColour.pyx", language="c++"),
)