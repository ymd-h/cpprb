FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends \
	build-essential \
	python-opengl \
	tk-dev \
	wget \
	x11-utils \
	xvfb \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip setuptools \
	&& pip3 install -U \
	coverage \
	cython \
	matplotlib \
	numpy \
	pyvirtualdisplay \
	sphinx \
	sphinx_rtd_theme \
	twine \
	wheel

COPY . /cpprb

WORKDIR /cpprb

RUN python3 setup.py build_ext --inplace --force --define CYTHON_TRACE_NOGIL \
	&& python3 setup.py bdist_wheel \
	&& pip3 install $(echo dist/cpprb-*.whl)['all'] \
	&& mkdir -p /tmp \
	&& mv cpprb/*.html cpprb/*.cpp /tmp/ \
	&& rm -rf /cpprb/* \
	&& mkdir -p /cpprb/public/annotation \
	&& mkdir -p /cpprb/cpp \
	&& mv /tmp/*.html /cpprb/public/annotation/ \
	&& mv /tmp/*.cpp /cpprb/cpp/

CMD ["/bin/bash"]
