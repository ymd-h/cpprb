FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends \
	libopenmpi-dev libgl1-mesa-dev zlib1g-dev \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install tensorflow==1.14 \
	&& pip install gym \
	&& pip install pandas ray[rllib] chainerrl perfplot \
	&& git clone https://github.com/openai/baselines.git \
	&& pip install ./baselines \
	&& rm -rf baselines


# OpenAI Baselines requires TensorFlow 1.14
# OpenAI Baselines at PyPI seems to be obsolete.
# gym (from cpprb) requires certain version of cloudpickle (install earlier)
# RLlib silently requires Pandas

CMD ["bash"]
