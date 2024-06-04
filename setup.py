import os
import platform
import sys
import sysconfig
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

debug = os.getenv("DEBUG_CPPRB")

# https://stackoverflow.com/a/73973555
on_CI = (  # noqa: N816
    os.getenv("ON_CI")
    or os.getenv("GITHUB_ACTIONS")
    or os.getenv("TRAVIS")
    or os.getenv("CIRCLECI")
    or os.getenv("GITLAB_CI")
)

arm_mac = (sysconfig.get_platform().split("-")[0] == "macosx") and (platform.machine() == "arm64")

requires = ["numpy"]
setup_requires = ["wheel"]

if sys.version_info >= (3, 9):
    # NumPy 2.0 breaks ABI compatibility.
    # To support both NumPy 1.x and 2.x, build with NumPy 2.x
    # cf. https://numpy.org/devdocs/dev/depending_on_numpy.html#numpy-2-abi-handling
    setup_requires.append("numpy>=2.0.0rc2")
elif (sys.version_info < (3, 9)) and not arm_mac:
    # NumPy 1.20 breaks ABI compatibility.
    # To support both NumPy 1.19.x and 1.20+, build with NumPy 1.19
    # This is necessary to work with old library which requires old NumPy.

    # NumPy 1.19.5 doesn't support Python 3.10+
    # NumPy 1.19.5 doesn't distribute wheel for macOS aarch64,
    setup_requires.append("numpy<1.20")
else:
    setup_requires.append("numpy")

rb_source = "cpprb/PyReplayBuffer"
cpp_ext = ".cpp"
pyx_ext = ".pyx"

extras = {
    "gym": ["matplotlib", "pyvirtualdisplay"],
    "api": ["sphinx", "sphinx_rtd_theme", "sphinx-automodapi"],
    "dev": ["coverage[toml]", "cython", "scipy", "twine", "unittest-xml-reporting"],
}

if sys.version_info < (3, 12):
    # ray doesn't support Python 3.12+, yet.
    # Issue: https://github.com/ray-project/ray/issues/40211
    extras["dev"].append("ray")

if platform.system() != "Windows":
    # jax doesn't support Windows
    extras["dev"].append("jax[cpu]")

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras["all"] = all_deps

# Set compiler flags depending on platform
if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17"]
    extra_link_args = None
    if debug:
        extra_compile_args.append("/DCYTHON_TRACE_NOGIL=1")
else:
    extra_compile_args = ["-std=c++17"]
    if (platform.system() != "Darwin") and not on_CI:
        # '-march=native' is not supported on Apple M1/M2 with clang
        # Ref: https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
        extra_compile_args.append("-march=native")

    extra_link_args = ["-std=c++17", "-pthread"]
    if debug:
        extra_compile_args.append("-DCYTHON_TRACE_NOGIL=1")

# Check cythonize or not
pyx_file = rb_source + pyx_ext
use_cython = os.path.exists(pyx_file)

if use_cython:
    suffix = pyx_ext
    setup_requires.extend(["cython>=3.0.10"])
    compiler_directives = {"language_level": "3"}

    if debug:
        compiler_directives["linetrace"] = True
else:
    suffix = cpp_ext

# Set ext_module
ext = [["cpprb", "PyReplayBuffer"], ["cpprb", "VectorWrapper"]]

ext_modules = [
    Extension(
        ".".join(e),
        sources=["/".join(e) + suffix],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    for e in ext
]


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

            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                compiler_directives=compiler_directives,
                include_path=["."],
                annotate=True,
            )
        super().finalize_options()


description = "ReplayBuffer for Reinforcement Learning written by C++ and Cython"
README = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
if os.path.exists(README):
    with open(README, encoding="utf-8") as f:
        long_description = f.read()
    long_description_content_type = "text/markdown"
else:
    warnings.warn("No README.md", stacklevel=2)
    long_description = description
    long_description_content_type = "text/plain"

setup(
    name="cpprb",
    author="Yamada Hiroyuki",
    description=description,
    version="11.0.0",
    install_requires=requires,
    setup_requires=setup_requires,
    extras_require=extras,
    cmdclass={"build_ext": LazyImportBuildExtCommand},
    url="https://ymd_h.gitlab.io/cpprb/",
    project_urls={
        "Source Code": "https://gitlab.com/ymd_h/cpprb",
        "Mirror": "https://github.com/ymd-h/cpprb",
        "Change Log": "https://ymd_h.gitlab.io/cpprb/changelog/",
        "Bug Report & QA": "https://github.com/ymd-h/cpprb/discussions",
    },
    ext_modules=ext_modules,
    include_dirs=["cpprb"],
    packages=["cpprb"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    long_description=long_description,
    long_description_content_type=long_description_content_type,
)
