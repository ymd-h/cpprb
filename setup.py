import os
from setuptools import setup,Extension,find_packages
from Cython.Build import cythonize
import numpy as np

if os.path.exists("cpprb/PyReplayBuffer.pyx"):
    ext_modules = cythonize([Extension("cpprb.ReplayBuffer",
                                       sources=["cpprb/PyReplayBuffer.pyx"],
                                       extra_compile_args=["-std=c++17",
                                                           "-march=native"],
                                       extra_link_args=["-std=c++17","-pthread"],
                                       language="c++"),
                             Extension("cpprb.VectorWrapper",
                                       sources=["cpprb/VectorWrapper.pyx"],
                                       extra_compile_args=["-std=c++17",
                                                           "-march=native"],
                                       extra_link_args=["-std=c++17","-pthread"],
                                       language="c++")],
                            compiler_directives={'language_level':"3"},
                            include_path=["."])
else:
    ext_modules = [Extension("cpprb.ReplayBuffer",
                             sources=["cpprb/PyReplayBuffer.cpp"],
                             extra_compile_args=["-std=c++17",
                                                 "-march=native"],
                             extra_link_args=["-std=c++17","-pthread"],
                             language="c++"),
                   Extension("cpprb.VectorWrapper",
                             sources=["cpprb/VectorWrapper.cpp"],
                             extra_compile_args=["-std=c++17",
                                                 "-march=native"],
                             extra_link_args=["-std=c++17","-pthread"],
                             language="c++")]

setup(name = "cpprb",
      author="Yamada Hiroyuki",
      author_email="incoming+ymd-h-cpprb-10328285-issue-@incoming.gitlab.com",
      description = "ReplayBuffer for Reinforcement Learning written by C++",
      version="7.5.0",
      install_requires=["cython>=0.29","numpy"],
      url="https://ymd_h.gitlab.io/cpprb/",
      ext_modules = ext_modules,
      include_dirs=["cpprb",np.get_include()],
      packages=["cpprb"],
      classifiers = ["Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                     "Development Status :: 4 - Beta",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Topic :: Software Development :: Libraries"],
      long_description = """cpprb is a python package written by C++.
The package provides replay buffer classes for reinforcement learning.

Complicated calculation (e.g. Segment Tree) are offloaded onto C++ which must be much faster than Python.

Internal C++ classes and corresponding Python wrapper classes share memory by implementing buffer protocol on cython to avoid overhead of copying large data.

This package requires C++17 compatible compiler to build.""")
