ARG arch="x86_64"

FROM iquiw/alpine-emacs AS README
WORKDIR /work
COPY README.org LICENSE .
RUN emacs --batch README.org --eval '(org-md-export-to-markdown)'


FROM quay.io/pypa/manylinux2014_${arch} AS manylinux
WORKDIR /work
COPY --from=README /work/README.md /work/README.md
COPY pyproject.toml setup.py LICENSE MANIFEST.in .
COPY cpprb cpprb/
ARG ON_CI
RUN ON_CI=${ON_CI} /opt/python/cp38-cp38/bin/pip wheel . -w /work/wheel --no-deps && \
    ON_CI=${ON_CI} /opt/python/cp39-cp39/bin/pip wheel . -w /work/wheel --no-deps && \
    ON_CI=${ON_CI} /opt/python/cp310-cp310/bin/pip wheel . -w /work/wheel --no-deps && \
    ON_CI=${ON_CI} /opt/python/cp311-cp311/bin/pip wheel . -w /work/wheel --no-deps && \
    ON_CI=${ON_CI} /opt/python/cp312-cp312/bin/pip wheel . -w /work/wheel --no-deps && \
    auditwheel repair /work/wheel/cpprb-*.whl -w /dist


FROM python:latest AS test
WORKDIR /work
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install hatch
COPY pyproject.tom setup.py LICENSE MANIFEST.in .
COPY cpprb cpprb/
RUN hatch env create test
COPY test test/
RUN hatch run test:run-cov && \
    hatch run cov:combine && \
    hatch run cov:report


FROM scratch AS results
COPY --from=manylinux /dist/cpprb-* /dist/
CMD [""]
