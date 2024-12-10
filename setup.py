import os
import platform
import sys
import sysconfig
import warnings

from setuptools import Extension, setup
import numpy as np

debug = os.getenv("DEBUG_CPPRB")

# https://stackoverflow.com/a/73973555
on_CI = (  # noqa: N816
    os.getenv("ON_CI")
    or os.getenv("GITHUB_ACTIONS")
    or os.getenv("TRAVIS")
    or os.getenv("CIRCLECI")
    or os.getenv("GITLAB_CI")
)


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



compiler_directives = {"language_level": "3"}
if debug:
    compiler_directives["linetrace"] = True


# Set ext_module
ext = [["cpprb", "PyReplayBuffer"], ["cpprb", "VectorWrapper"]]

ext_modules = [
    Extension(
        ".".join(e),
        sources = ["src/" + "/".join(e) + ".pyx"],
        include_dirs = [np.get_include()],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        language = "c++",
        compiler_directives = compiler_directives,
        cython_include_dirs = ["."],
    )
    for e in ext
]


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
    description=description,
    ext_modules=ext_modules,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
)
