FROM python:3.8

RUN apt update \
    && apt install -y --no-install-recommends \
    build-essential \
    graphviz \
    python3-opengl \
    tk-dev \
    x11-utils \
    xvfb \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip setuptools \
    && pip3 install -U \
    coverage \
    cython \
    gym[box2d] \
    gym-algorithmic \
    gym-legacy-toytext \
    jax[cpu] \
    matplotlib \
    'numpy<1.20.0' \
    pyvirtualdisplay \
    ray \
    sphinx \
    sphinx-automodapi \
    sphinx_rtd_theme \
    twine \
    unittest-xml-reporting \
    wheel

CMD ["/bin/bash"]
