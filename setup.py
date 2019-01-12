from distutils.core import setup,Extension
from Cython.Build import cythonize

ext_modules = cythonize(Extension("ReplayBuffer",
                                  sources=["ReplayBuffer.pyx"],
                                  extra_compile_args=["-std=c++17"],
                                  extra_link_args=["-std=c++17"],
                                  language="c++"),
                        compiler_directives={'language_level':"3"})

setup(ext_modules = ext_modules,include_dirs=["."])
