from distutils.core import setup,Extension
from Cython.Build import cythonize

ext_modules = cythonize([Extension("SegmentTree",
                                   sources=["SegmentTree.pyx"],
                                   extra_compile_args=["-std=c++17"],
                                   extra_link_args=["-std=c++17"],
                                   language="c++17"),
                         Extension("ReplayBuffer",
                                   sources=["ReplayBuffer.pyx"],
                                   extra_compile_args=["-std=c++17"],
                                   extra_link_args=["-std=c++17"],
                                   language="c++")],
                         compiler_directives={'language_level':"3"},
                         include_path=["."])

setup(ext_modules = ext_modules,include_dirs=["."])
