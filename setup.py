import os
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

debug = os.getenv('DEBUG_CPPRB')

requires = ["numpy"]

extras = {
    'gym': ["matplotlib", "pyvirtualdisplay"]
}
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

if platform.system() == 'Windows':
    extra_compile_args = ["/std:c++17"]
    extra_link_args = None
    if debug:
        extra_compile_args.append('/DCYTHON_TRACE_NOGIL=1')
else:
    extra_compile_args = ["-std=c++17","-march=native"]
    extra_link_args = ["-std=c++17", "-pthread"]
    if debug:
        extra_compile_args.append('-DCYTHON_TRACE_NOGIL=1')

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
       ["cpprb","VectorWrapper"]]

ext_modules = wrap([Extension(".".join(e),
                              sources=["/".join(e) + suffix],
                              extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args,
                              language="c++") for e in ext])


class LazyImportBuildExtCommand(build_ext):
    """
    build_ext command class for lazy numpy import
    """
    def run(self):
        import numpy as np

        self.include_dirs.append(np.get_include())

        build_ext.run(self)

setup(name="cpprb",
      author="Yamada Hiroyuki",
      author_email="incoming+ymd-h-cpprb-10328285-issue-@incoming.gitlab.com",
      description="ReplayBuffer for Reinforcement Learning written by C++",
      version="8.1.3",
      install_requires=requires,
      setup_requires=["numpy"],
      extras_require=extras,
      cmdclass={'build_ext': LazyImportBuildExtCommand},
      url="https://ymd_h.gitlab.io/cpprb/",
      ext_modules=ext_modules,
      include_dirs=["cpprb"],
      packages=["cpprb", "cpprb.gym"],
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
