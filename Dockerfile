FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends \
	build-essential \
	python-opengl \
	tk-dev \
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

CMD ["/bin/bash"]
