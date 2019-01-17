from setuptools import setup,Extension,find_packages
from Cython.Build import cythonize

ext_modules = cythonize([Extension("ymd_K.SegmentTree",
                                   sources=["ymd_K/PySegmentTree.pyx"],
                                   extra_compile_args=["-std=c++17"],
                                   extra_link_args=["-std=c++17"],
                                   language="c++"),
                         Extension("ymd_K.ReplayBuffer",
                                   sources=["ymd_K/PyReplayBuffer.pyx"],
                                   extra_compile_args=["-std=c++17"],
                                   extra_link_args=["-std=c++17"],
                                   language="c++")],
                         compiler_directives={'language_level':"3"},
                         include_path=["."])

setup(name = "ymd_K", ext_modules = ext_modules, include_dirs=["ymd_K"],
      packages=["ymd_K"])
