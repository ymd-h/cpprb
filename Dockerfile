FROM gcc:latest

ENV PYTHON_VERSION 3.8.0

RUN apt update \
	&& apt install -y --no-install-recommends \
	build-essential \
	python-opengl \
	tk-dev \
	wget \
	x11-utils \
	xvfb \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/* \
	&& wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
	&& tar -xf Python-${PYTHON_VERSION}.tar.xz \
	&& cd Python-${PYTHON_VERSION} \
	&& ./configure --enable-shared --with-ensurepip --enable-optimizations \
	&& make -s -j "$(nproc)" \
	&& make install \
	&& /sbin/ldconfig -v \
	&& cd / \
	&& rm -rf Python-${PYTHON_VERSION}{,.tar.xz}

RUN pip3 install -U pip setuptools \
	&& pip3 install -U \
	coverage \
	cython \
	matplotlib \
	numpy \
	pyvirtualdisplay \
	sphinx \
	twine \
	wheel

COPY . /cpprb

WORKDIR /cpprb

RUN python3 setup.py build_ext --inplace --force --define CYTHON_TRACE_NOGIL \
	&& python3 setup.py sdist bdist_wheel \
	&& pip3 install $(echo dist/cpprb-*.whl)['all'] \
	&& mkdir -p /tmp \
	&& mv dist/cpprb-*.whl /tmp/ \
	&& mv cpprb/*.html /tmp/ \
	&& rm -rf /cpprb/* \
	&& mkdir -p /cpprb/dist \
	&& mkdir -p /cpprb/public/annotation \
	&& mv /tmp/cpprb-*.whl /cpprb/dist/ \
	&& mv /tmp/*.html /cpprb/public/annotation/

CMD ["/bin/bash"]
