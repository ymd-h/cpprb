import os
from setuptools import setup, Extension, find_packages
import numpy as np


requires = ["numpy"]

extras = {
    'gym': ["matplotlib", "pyvirtualdisplay"],
    'rl': ["scipy","tf-nightly-2.0-preview"]
}
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps


if os.path.exists("cpprb/PyReplayBuffer.pyx"):
    from Cython.Build import cythonize

    suffix = ".pyx"
    wrap = lambda x: cythonize(x,
                               compiler_directives={'language_level': "3"},
                               include_path=["."],
                               annotate=True)
    requires.extend(["cython>=0.29"])
else:
    suffix = ".cpp"
    wrap = lambda x: x

ext = [["cpprb","PyReplayBuffer"],
       ["cpprb","VectorWrapper"],
       ["cpprb","experimental","PyReplayBuffer"]]

ext_modules = wrap([Extension(".".join(e),
                              sources=["/".join(e) + suffix],
                              extra_compile_args=["-std=c++17",
                                                  "-march=native"],
                              extra_link_args=["-std=c++17", "-pthread"],
                              language="c++") for e in ext])

setup(name="cpprb",
      author="Yamada Hiroyuki",
      author_email="incoming+ymd-h-cpprb-10328285-issue-@incoming.gitlab.com",
      description="ReplayBuffer for Reinforcement Learning written by C++",
      version="7.10.7",
      install_requires=requires,
      extras_require=extras,
      url="https://ymd_h.gitlab.io/cpprb/",
      ext_modules=ext_modules,
      include_dirs=["cpprb", np.get_include()],
      packages=["cpprb", "cpprb.gym","cpprb.rl","cpprb.experimental"],
      classifiers=["Programming Language :: Python",
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   "Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Software Development :: Libraries"],
      long_description="""cpprb is a python package written by C++.
The package provides replay buffer classes for reinforcement learning.

Complicated calculation (e.g. Segment Tree) are offloaded onto C++ which must be much faster than Python.

Internal C++ classes and corresponding Python wrapper classes share memory by implementing buffer protocol on cython to avoid overhead of copying large data.

This package requires C++17 compatible compiler to build.""")
