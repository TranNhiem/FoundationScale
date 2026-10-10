ARG PY=3.12
FROM python:${PY}-slim
RUN pip install -q pytest
ENV FOO=bar
WORKDIR /testbed
COPY repo/ /testbed
