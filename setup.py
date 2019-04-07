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
      version="7.4.0",
      install_requires=["cython>=0.29","numpy"],
      url="https://ymd_h.gitlab.io/cpprb/",
      ext_modules = ext_modules,
      include_dirs=["cpprb",np.get_include()],
      packages=["cpprb"],
      classifiers = ["Programming Language :: Python",
                     "Programming Language :: Python :: 3"])
