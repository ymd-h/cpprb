#!/bin/bash

if [[ "root" == $(whoami) ]]; then
SUDO=""
else
SUDO=sudo
fi

${SUDO} pip-${1} uninstall -y cpprb
${SUDO} rm -rf dist build cpprb.egg-info
rm cpprb/{{,experimental/}PyReplayBuffer.cpp,VectorWrapper.cpp}
python${1} setup.py clean && python${1} setup.py build_ext --inplace --force --define CYTHON_TRACE_NOGIL && python${1} setup.py build && ${SUDO} python${1} setup.py install

