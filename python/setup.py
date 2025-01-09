from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "python.mcts.c",
        sources=["python/mcts/c.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="cy",
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=True),
)