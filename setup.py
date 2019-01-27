from setuptools import setup,Extension,find_packages
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize([Extension("ymd_K.ReplayBuffer",
                                   sources=["ymd_K/PyReplayBuffer.pyx"],
                                   extra_compile_args=["-std=c++17"],
                                   extra_link_args=["-std=c++17"],
                                   language="c++")],
                         compiler_directives={'language_level':"3"},
                         include_path=["."])

setup(name = "ymd_K",
      version="4.0.0",
      ext_modules = ext_modules,
      include_dirs=["ymd_K",np.get_include()],
      packages=["ymd_K"])
