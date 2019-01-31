from setuptools import setup,Extension,find_packages
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize([Extension("cpprb.ReplayBuffer",
                                   sources=["cpprb/PyReplayBuffer.pyx"],
                                   extra_compile_args=["-std=c++17"],
                                   extra_link_args=["-std=c++17"],
                                   language="c++")],
                         compiler_directives={'language_level':"3"},
                         include_path=["."])

setup(name = "cpprb",
      version="5.2.3",
      ext_modules = ext_modules,
      include_dirs=["cpprb",np.get_include()],
      packages=["cpprb"])
