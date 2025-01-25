from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "mcts.c", # prepend "python." to the module name if needed
        sources=["python/mcts/c.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "selfplay.c", # prepend "python." to the module name if needed
        sources=["python/selfplay/c.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="cy",
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=False),
)