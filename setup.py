import os
import platform
import warnings
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

debug = os.getenv('DEBUG_CPPRB')

requires = ["numpy"]
setup_requires = ["numpy"]

rb_source = "cpprb/PyReplayBuffer"
cpp_ext = ".cpp"
pyx_ext = ".pyx"

extras = {
    'gym': ["matplotlib", "pyvirtualdisplay"],
    'api': ["sphinx","sphinx_rtd_theme","sphinx-automodapi"],
    'dev': ["coverage","cython","gym[box2d]","twine","unittest-xml-reporting","wheel"]
}
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

# Set compiler flags depending on platform
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

# Check cythonize or not
cpp_file = rb_source + cpp_ext
pyx_file = rb_source + pyx_ext
use_cython = (not os.path.exists(cpp_file)
              or (os.path.exists(pyx_file)
                  and (os.path.getmtime(cpp_file) < os.path.getmtime(pyx_file))))
if use_cython:
    suffix = pyx_ext
    setup_requires.extend(["cython>=0.29"])
    compiler_directives = {'language_level': "3"}

    if debug:
        compiler_directives['linetrace'] = True
else:
    suffix = cpp_ext

# Set ext_module
ext = [["cpprb","PyReplayBuffer"],
       ["cpprb","VectorWrapper"]]

ext_modules = [Extension(".".join(e),
                         sources=["/".join(e) + suffix],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         language="c++") for e in ext]

class LazyImportBuildExtCommand(build_ext):
    """
    build_ext command class for lazy numpy and cython import
    """
    def run(self):
        import numpy as np

        self.include_dirs.append(np.get_include())
        build_ext.run(self)

    def finalize_options(self):
        if use_cython:
            from Cython.Build import cythonize
            self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                                                      compiler_directives=compiler_directives,
                                                      include_path=["."],
                                                      annotate=True)
        super().finalize_options()


description = "ReplayBuffer for Reinforcement Learning written by C++ and Cython"
README = os.path.join(os.path.abspath(os.path.dirname(__file__)),'README.md')
if os.path.exists(README):
    with open(README,encoding='utf-8') as f:
        long_description = f.read()
    long_description_content_type='text/markdown'
else:
    warnings.warn("No README.md")
    long_description =  description
    long_description_content_type='text/plain'

setup(name="cpprb",
      author="Yamada Hiroyuki",
      description=description,
      version="9.3.1",
      install_requires=requires,
      setup_requires=setup_requires,
      extras_require=extras,
      cmdclass={'build_ext': LazyImportBuildExtCommand},
      url="https://ymd_h.gitlab.io/cpprb/",
      ext_modules=ext_modules,
      include_dirs=["cpprb"],
      packages=["cpprb"],
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
      long_description=long_description,
      long_description_content_type=long_description_content_type)
