FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends libopenmpi-dev zlib1g-dev \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install tf-nightly==2.3.0.dev20200604 dm-reverb-nightly==0.1.0.dev20200616 perfplot


# Reverb requires development version TensorFlow

CMD ["bash"]
